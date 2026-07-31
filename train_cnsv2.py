"""CNSv2 full training on a BlenderProc-rendered dataset (fp32, with checkpoints).

    python3 train_cnsv2.py --data data/bproc_train --iters 40000 --out checkpoints/cnsv2.pth

Loads BlenderProc scenes (bproc_gen.py output), labels them with the PBVS
supervisor (load_bproc_samples), precomputes the frozen ViT features once, then
trains the refine->prob-match->controller head. Offline supervised (no DAgger
yet). Saves periodic + final checkpoints with the canonical constants the
laptop-side denormalization needs (CNSv2_5090_SETUP.md Sec 8).
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from cns.models.cnsv2_net import build_model
from cns.sim.cnsv2_data import (load_bproc_samples, precompute_backbone_features,
                                build_feature_cache)

CONFIG = {"K": 16, "feat_dim": 768, "intrinsic": {"fx": 512, "fy": 512, "cx": 256,
          "cy": 256, "H1": 512, "W1": 512},
          # NOT a runtime value -- nothing reads it. The paper's Table I env uses
          # d* = 1.0 m, but our sampler draws r in [0.5, 0.9] (CNS-v1 values), so the
          # d* actually baked into labels is per-scene: measured 0.503-0.940 m,
          # median 0.721. Recorded honestly because a deployment consumer reading
          # config["d_star"] and getting 1.0 would mis-scale every translation by up
          # to 2x. The real scale is computed per scene as ||tPo|| at the desired pose.
          "d_star_paper": 1.0,
          "d_star_sampled_range": [0.5, 0.9],
          "d_star_note": "computed per-scene at runtime; see cns/sim/supervisor.py"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BlenderProc scene dir (scene_*.npz)")
    ap.add_argument("--iters", type=int, default=40000)
    # Paper Sec. IV-A trains at batch 16 for ~40k iterations. The 4060 box ran 8
    # because 8GB of VRAM could not hold more; 32GB per 5090 can.
    ap.add_argument("--batch", type=int, default=16)
    # Paper Sec. I: "we employ mixed floating point training and inference so the
    # model runs in real-time". bf16 rather than fp16 because the feature
    # correlation overflows fp16's 65504 range; prob_match.py additionally forces
    # score_matrix and particle-to-grid to fp32 regardless of the autocast dtype.
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--fine-dim", type=int, default=128,
                    help="fine-grained CNN branch width (Fig. 2); 0 disables it")
    ap.add_argument("--val-batch", type=int, default=8,
                    help="val forward chunk size; full-split at once OOMs on 8GB")
    # 3e-4 COLLAPSES this head onto the constant (mean-velocity) solution: output
    # becomes identical for every input (pairwise cos 1.0000) and l_dir sticks at
    # the best-constant baseline. Measured at N=64: lr 3e-4 -> l_dir 0.653 (== const
    # 0.653); lr 5e-5 -> 0.016. Same at N=256: 0.702 (== const 0.698) vs 0.120.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="checkpoints/cnsv2.pth")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    # Down-weight the magnitude term in `l_dir + w*l_norm`. Needed once near-goal
    # data is present: sigma_inv targets reach -3.9, so l_norm runs ~1.09 against
    # l_dir ~0.69 -- ~60% of the gradient goes to magnitude while direction, which
    # is what actually drives servo convergence, stalls.
    ap.add_argument("--norm-weight", type=float, default=1.0)
    # Drop samples whose pose error is below the correspondence grid's spatial
    # resolution. Patches are 512/32 = 16px, so ||vel_si||~0.03 is ~0.5 patch of
    # image displacement: measured paired cos(Fc,Fd) = 0.93 there, i.e. the two
    # views are nearly identical in feature space and the label is unlearnable.
    # Worse, sigma_inv(0.007) = -4.15 vs a median target of 1.49, so those 11% of
    # samples contributed 32% of the L1 magnitude loss -- an unfittable term that
    # dominated the gradient and stalled direction learning entirely
    # (16k-iter run: l_dir flat at 0.69 for 44 validations, l_norm stuck ~1.05).
    # 0.05 keeps the USEFUL near-goal band (~1.2-6 patches) and drops the rest.
    # Stratified batch sampling over ||vel_si|| bins. DAgger rollout states are
    # inherently near-goal-heavy -- expert convergence is geometric, so most steps
    # of a converging trajectory sit near the goal (measured: 79.4% of round-0
    # rollouts had ||vel_si||<0.5). Aggregating those onto a far-heavy uniform set
    # drifted the mix 1.8% -> 41% -> 56.3% near-goal across rounds, and the
    # closed-loop TE ratio got WORSE (1.350 -> 1.569) even as offline l_dir
    # improved 2% -> 64% over baseline: the model gained near-goal skill and lost
    # the far field, which is where every episode starts. The paper avoids this by
    # running uniform and DAgger sampling CONCURRENTLY (Fig. 3) so the database
    # stays balanced; we aggregate, so we rebalance here instead. Reweights rather
    # than discards, so no data is thrown away.
    ap.add_argument("--balance", action="store_true",
                    help="stratified batch sampling over ||vel_si|| bins")
    ap.add_argument("--min-vel", type=float, default=0.0,
                    help="drop samples with ||vel_si|| below this (sub-patch, unlearnable)")
    # This head overfits 1850 pairs by ~iter 1000 (val l_dir bottoms at 0.596 then
    # climbs back above the best-constant baseline by iter 2000 while train keeps
    # falling). Without best-tracking the FINAL checkpoint is the worst one, so
    # track the best val l_dir and stop when it stops improving.
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many validations with no val l_dir gain (0=off)")
    # ---- concurrent DAgger (paper Fig. 3) --------------------------------
    # Not a phase 2. The collector runs at the same time as this process, reading
    # weights from --sync-path; we ingest whatever rollouts have landed and mix
    # them in at a FIXED fraction. The fixed fraction is the point: aggregating
    # whole DAgger rounds previously drifted the mix to 56% near-goal and made the
    # closed loop worse (see cns/sim/dagger_pool.py).
    ap.add_argument("--dagger-dir", default="",
                    help="dir the DAgger collector writes scene_*.npz into")
    ap.add_argument("--dagger-frac", type=float, default=0.5,
                    help="fraction of each batch drawn from DAgger rollouts")
    ap.add_argument("--init", default="",
                    help="warm-start weights from this checkpoint (state_dict only; "
                         "optimizer, LR schedule and iteration counter start fresh). "
                         "Ignored if <out>_iter*.pth exist, so a resume always wins.")
    ap.add_argument("--feat-cache", default="",
                    help="reuse an existing uniform-half feature cache DIRECTORY "
                         "instead of <out>_feat. The cache depends only on --data, "
                         "so a second run should share it rather than spend ~20 min "
                         "and 39GB rebuilding an identical one.")
    ap.add_argument("--dagger-reserve", type=int, default=20000,
                    help="pre-allocated DAgger pair rows")
    ap.add_argument("--dagger-min-tv", type=float, default=0.015,
                    help="DAgger pool floor on TRANSLATION (unit-world). A pair is "
                         "dropped only if translation AND rotation are both below "
                         "their floors -- the old single --dagger-min-vel on the "
                         "mixed 6-vector norm cut 81%% of sub-20mm pairs, keeping "
                         "the ones with rotation and discarding aligned-but-offset "
                         "ones (Sec. 6.22). Default = half a 16px patch.")
    ap.add_argument("--dagger-min-rw", type=float, default=0.012,
                    help="DAgger pool floor on ROTATION (rad); half a patch of "
                         "image motion at the canonical intrinsics.")
    ap.add_argument("--ingest-every", type=int, default=200,
                    help="iters between scans of --dagger-dir")
    ap.add_argument("--sync-path", default="",
                    help="where to publish weights for the collector "
                         "(default <out>_sync.pth when --dagger-dir is set)")
    ap.add_argument("--sync-every", type=int, default=500,
                    help="iters between weight publications (Fig. 3 'Synchronize "
                         "Weights Periodically')")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[load] {args.data}", flush=True)
    net = build_model(K=CONFIG["K"], refine_layers=4, ctrl_dim=256,
                      fine_dim=args.fine_dim).to(dev)
    CONFIG["fine_dim"] = args.fine_dim

    # Frozen-ViT feature cache, STREAMED to on-disk .npy memmaps.
    # The previous single-.pt cache could not scale: at 7115 pairs it held
    # Ic+Id (11.2GB uint8) + Fc+Fd (11.2GB each fp16) for a ~28GB peak and was
    # OOM-killed (exit 137) on this 31GB box, and torch.save() additionally needs
    # every byte resident to serialise. build_feature_cache() streams one scene at
    # a time (peak ~64MB) and training then demand-pages 8 rows per step.
    # Derived from --out by default, but overridable: the uniform half of the
    # database has nothing to do with which run is training on it, and rebuilding
    # it costs ~20 min and 39GB. A second run with a different --out would
    # otherwise silently rebuild an identical cache.
    cache = args.feat_cache or args.out.replace(".pth", "_feat")   # a DIRECTORY
    import glob as _g
    n_scenes = len(_g.glob(os.path.join(args.data, "scene_*.npz")))
    meta_p = os.path.join(cache, "meta.pt")
    if not os.path.exists(meta_p):
        build_feature_cache(args.data, net, dev, cache, val_frac=args.val_frac)
    meta = torch.load(meta_p, map_location="cpu")
    # Scene-count guard: adding scenes then reusing a stale cache would silently
    # train on the old sample set (this bit us once).
    if meta.get("n_scenes") != n_scenes:
        raise SystemExit(
            f"[feat] STALE CACHE: {cache} was built from {meta.get('n_scenes')} scenes "
            f"but {args.data} now has {n_scenes}. Delete it and rerun:\n    rm -rf {cache}")
    Fc = np.load(os.path.join(cache, "fc.npy"), mmap_mode="r")
    Fd = np.load(os.path.join(cache, "fd.npy"), mmap_mode="r")
    # Raw images for the Fig. 2 fine CNN branch. A cache built before that branch
    # existed has no ic.npy, and silently training without the branch would be the
    # kind of quiet fidelity regression this repo has been bitten by before.
    ic_p = os.path.join(cache, "ic.npy")
    if args.fine_dim > 0 and not os.path.exists(ic_p):
        raise SystemExit(
            f"[feat] cache {cache} predates the fine-CNN branch (no ic.npy).\n"
            f"    rm -rf {cache}    and rerun, or pass --fine-dim 0 for the "
            f"coarse-only ablation.")
    Ic = np.load(ic_p, mmap_mode="r") if args.fine_dim > 0 else None
    Id = np.load(os.path.join(cache, "id.npy"), mmap_mode="r") if args.fine_dim > 0 else None
    vsi, tr, va = meta["vsi"], meta["tr"], meta["va"]
    # Refuse a leaky (pair-level) split: it reports memorization as generalization.
    _sid = meta.get("scene_id")
    if _sid is not None and len(va):
        _leak = len(set(_sid[va].tolist()) & set(_sid[tr].tolist())) / len(set(_sid[va].tolist()))
        if _leak > 0.01:
            raise SystemExit(
                f"[feat] LEAKY CACHE: {100*_leak:.0f}% of val scenes also appear in "
                f"train. This cache was built with the old pair-level split, whose "
                f"val l_dir measures memorization (measured 0.038 seen vs 0.243 "
                f"unseen on the same checkpoint).\n    rm -rf {cache}   and rerun.")
    # fd.npy has ONE row per scene (the desired view is shared by every pair in a
    # scene); scene_id maps pair -> scene. Saves ~16x on fd.
    scene_id = meta.get("scene_id")
    print(f"[feat] cache {cache}: {Fc.shape[0]} pairs ({n_scenes} scenes), "
          f"train {len(tr)} / val {len(va)}", flush=True)

    if args.min_vel > 0:
        keep = (vsi.norm(dim=-1) >= args.min_vel)
        n_tr0, n_va0 = len(tr), len(va)
        tr, va = tr[keep[tr]], va[keep[va]]
        print(f"[filter] --min-vel {args.min_vel}: train {n_tr0}->{len(tr)} "
              f"({100*(1-len(tr)/n_tr0):.1f}% dropped), val {n_va0}->{len(va)}", flush=True)
        nn = vsi[tr].norm(dim=-1)
        print(f"[filter] remaining ||vel_si||: min {nn.min():.4f} median {nn.median():.3f}; "
              f"near-goal(<0.5) now {100*(nn < 0.5).float().mean():.1f}% of train", flush=True)

    # ---- concurrent DAgger pool ----------------------------------------
    pool = sync_path = None
    if args.dagger_dir:
        from cns.sim.dagger_pool import DaggerPool
        H16, W16, C = Fc.shape[1], Fc.shape[2], Fc.shape[3]
        H1, W1 = meta.get("img_shape", (512, 512, 3))[:2]
        pool = DaggerPool(args.out.replace(".pth", "_dagger"), H16, W16, C,
                          H1=H1, W1=W1, reserve_pairs=args.dagger_reserve,
                          reserve_scenes=max(1, args.dagger_reserve // 5),
                          min_tv=args.dagger_min_tv,
                          min_rw=args.dagger_min_rw)
        os.makedirs(args.dagger_dir, exist_ok=True)
        sync_path = args.sync_path or args.out.replace(".pth", "_sync.pth")
        print(f"[dagger] pool {pool.dir}: {pool.stats()}", flush=True)
        print(f"[dagger] publishing weights to {sync_path} every "
              f"{args.sync_every} iters; mixing {args.dagger_frac:.0%} of each batch",
              flush=True)

    samp_w = None
    if args.balance:
        edges = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0])
        nrm = vsi.norm(dim=-1)[tr]
        b = torch.bucketize(nrm, edges)
        cnt = torch.bincount(b, minlength=len(edges) + 1).float().clamp_min(1.0)
        samp_w = (1.0 / cnt[b])
        samp_w = samp_w / samp_w.sum()
        share = (cnt / cnt.sum() * 100)
        print(f"[balance] train bin counts {cnt.long().tolist()} "
              f"(= {[f'{x:.1f}%' for x in share.tolist()]}) -> equalised by resampling",
              flush=True)

    decay, no_decay = net.get_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    # --- resume from the latest checkpoint if present (survives job kills) ---
    import glob as _glob
    start_it = 0
    cks = sorted(_glob.glob(args.out.replace(".pth", "_iter*.pth")),
                 key=lambda p: int(p.split("iter")[-1].split(".")[0]))
    if cks:
        ck = torch.load(cks[-1], map_location=dev)
        net.load_state_dict(ck["state_dict"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "sched" in ck: sched.load_state_dict(ck["sched"])
        start_it = ck.get("iters_done", 0)
        print(f"[resume] {cks[-1]} @ iter {start_it}", flush=True)
    elif args.init:
        # WARM START, not a resume: weights only, so the LR schedule, optimizer
        # state and iteration counter all begin fresh. This is what a DAgger round 2
        # wants -- carry the policy forward so its rollouts are meaningful, but
        # train on a new database from iteration 0. Deliberately in the `elif`: a
        # real resume checkpoint must always win over --init, or a restarted job
        # would silently throw away its own progress.
        ick = torch.load(args.init, map_location=dev)
        # NON-STRICT, and it REPORTS. Adding RoPE to the controller renamed its
        # self-attention parameters (nn.MultiheadAttention in_proj/out_proj ->
        # MHAttention q/k/v/proj), so a strict load would just crash and a silent
        # non-strict load would hide that the whole controller had been re-initialized.
        # Print it, and refuse if almost nothing matched -- that means the wrong
        # checkpoint, not an intended architecture change.
        missing, unexpected = net.load_state_dict(ick["state_dict"], strict=False)
        own = set(net.state_dict().keys())
        loaded = len(own) - len(missing)
        print(f"[init] warm-started from {args.init} "
              f"(trained {ick.get('iters_done')} iters): {loaded}/{len(own)} tensors "
              f"loaded, {len(missing)} re-initialized, {len(unexpected)} unused",
              flush=True)
        if missing:
            groups = sorted({k.split(".")[0] + "." + k.split(".")[1]
                             for k in missing if "." in k})
            print(f"[init] RE-INITIALIZED (fresh weights): {', '.join(groups[:8])}"
                  + (" ..." if len(groups) > 8 else ""), flush=True)
        if loaded < 0.5 * len(own):
            raise SystemExit(
                f"[init] REFUSING: only {loaded}/{len(own)} tensors matched. That is "
                f"a checkpoint/architecture mismatch, not a warm start.")
        print(f"[init] optimizer/schedule/iter counter start fresh", flush=True)

    def _take(arr, idx):
        """Fc/Fd/Ic/Id are numpy memmaps; fancy-indexing copies just the needed rows."""
        return torch.from_numpy(np.ascontiguousarray(arr[idx.cpu().numpy()]))

    import contextlib
    _amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    def amp_ctx():
        if _amp_dtype is None or dev != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=_amp_dtype)

    # fp16 needs loss scaling; bf16 has fp32's exponent range and does not.
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    def _gather(fcA, fdA, icA, idA, sid, idx):
        fc = _take(fcA, idx).to(dev).float()
        didx = idx if sid is None else sid[idx]
        fd = _take(fdA, didx).to(dev).float()
        ic = idi = None
        if icA is not None:
            # [B,H,W,3] uint8 -> [B,3,H,W] float in [0,1], scaled on the GPU so the
            # host->device copy moves a quarter of the bytes.
            ic = _take(icA, idx).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
            idi = _take(idA, didx).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        return fc, fd, ic, idi

    def batch_forward(idx, didx_pool=None):
        """idx indexes the uniform cache; didx_pool (optional) indexes the DAgger
        pool. Both halves are concatenated into one forward so the batch statistics
        (and the LayerNorms) see the mixed distribution, not alternating ones."""
        parts = []
        if len(idx):
            parts.append(_gather(Fc, Fd, Ic, Id, scene_id, idx))
        if didx_pool is not None and len(didx_pool):
            parts.append(_gather(pool.fc, pool.fd,
                                 pool.ic if Ic is not None else None,
                                 pool.id, pool.scene_id, didx_pool))
        if len(parts) == 1:
            fc, fd, ic, idi = parts[0]
        else:
            fc = torch.cat([p[0] for p in parts])
            fd = torch.cat([p[1] for p in parts])
            ic = None if parts[0][2] is None else torch.cat([p[2] for p in parts])
            idi = None if parts[0][3] is None else torch.cat([p[3] for p in parts])
        return net.head_forward(fc, fd, Ic=ic, Id=idi)

    def batch_targets(idx, didx_pool=None):
        ts = []
        if len(idx):
            ts.append(vsi[idx])
        if didx_pool is not None and len(didx_pool):
            ts.append(pool.vsi[didx_pool])
        return torch.cat(ts).to(dev)

    @torch.no_grad()
    def validate():
        """Chunked over the val split. A single full-split forward OOMs on 8GB
        (185 val samples x 1024 tokens of cross-attention ~ 6GB of attention
        alone). The objectives are batch means, so a size-weighted mean over
        chunks reproduces the full-split number exactly."""
        net.eval()
        acc, seen = {}, 0
        for i in range(0, len(va), args.val_batch):
            chunk = va[i:i + args.val_batch]
            with amp_ctx():
                raw = batch_forward(chunk)
            res, _ = net.objectives(raw, vsi[chunk].to(dev), args.norm_weight)
            for k, v in res.items():
                acc[k] = acc.get(k, 0.0) + v * len(chunk)
            seen += len(chunk)
        net.train()
        return {k: v / seen for k, v in acc.items()}

    def save_to(path):
        torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "iters_done": it + 1,
                    "best_l_dir": best["l_dir"], "best_it": best["it"]}, path)

    best_path = args.out.replace(".pth", "_best.pth")
    best = {"l_dir": float("inf"), "it": -1}
    stale = 0
    # Carry the best-so-far across a resume. Without this, a resumed run starts at
    # inf and its FIRST validation always looks like an improvement, overwriting
    # cnsv2_best.pth with a worse model and silently losing the real optimum.
    if cks and "best_l_dir" in ck:
        best["l_dir"] = float(ck["best_l_dir"]); best["it"] = int(ck.get("best_it", -1))
        print(f"[resume] carrying best val l_dir {best['l_dir']:.4f} @ iter {best['it']}",
              flush=True)
    print(f"[train] {args.iters} iters (from {start_it}), batch {args.batch}, "
          f"amp={args.amp}, fine_dim={args.fine_dim}", flush=True)
    if pool is not None and not os.path.exists(sync_path):
        # The collector needs a policy before it can produce anything on-policy,
        # so publish once up front rather than making it wait --sync-every iters.
        torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                    "iters_done": start_it}, sync_path)
        print(f"[dagger] published initial weights -> {sync_path}", flush=True)

    net.train(); t0 = time.time(); hist = []
    for it in range(start_it, args.iters):
        # --- Fig. 3: ingest whatever the concurrent collector has produced ---
        if pool is not None and (it + 1) % args.ingest_every == 0:
            added = pool.ingest(args.dagger_dir, net.backbone, dev,
                                prune_ingested=True)
            if added:
                print(f"  [dagger] +{added} pairs -> {pool.stats()}", flush=True)

        # --- Fig. 3: publish weights for the collector to pick up ---
        if pool is not None and (it + 1) % args.sync_every == 0:
            tmp = sync_path + ".tmp"
            torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                        "iters_done": it + 1}, tmp)
            os.replace(tmp, sync_path)      # atomic: the collector may be reading

        # DAgger share of the batch, capped by what the pool actually holds.
        n_d = 0
        if pool is not None and pool.n_pairs > 0:
            n_d = min(int(round(args.batch * args.dagger_frac)), pool.n_pairs)
        n_u = args.batch - n_d
        if samp_w is not None:
            idx = tr[torch.multinomial(samp_w, n_u, replacement=True)] if n_u else tr[:0]
        else:
            idx = tr[torch.randint(0, len(tr), (n_u,))] if n_u else tr[:0]
        # pool.sample, not randint(0, n_pairs): eviction leaves the live rows a
        # scattered subset of the reserve, so a prefix draw would hit dead slots.
        didx = (pool.sample(n_d) if n_d else None)

        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            raw = batch_forward(idx, didx)
        # objectives() upcasts to fp32 internally, so the loss is computed in fp32
        # regardless of the autocast dtype.
        _, loss = net.objectives(raw, batch_targets(idx, didx), args.norm_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        scaler.step(opt); scaler.update(); sched.step()
        hist.append(float(loss.detach()))
        if (it + 1) % args.log_every == 0:
            v = validate()
            rate = (it + 1) / (time.time() - t0)
            if v["l_dir"] < best["l_dir"] - 1e-4:
                best.update(l_dir=v["l_dir"], it=it + 1); stale = 0
                save_to(best_path); mark = "  *best"
            else:
                stale += 1; mark = ""
            print(f"  iter {it+1:6d}/{args.iters}  train {np.mean(hist[-args.log_every:]):.4f}  "
                  f"val l_dir {v['l_dir']:.4f} l_norm {v['l_norm']:.4f}  "
                  f"{rate:.1f} it/s{mark}", flush=True)
            if args.patience and stale >= args.patience:
                print(f"[early-stop] {stale} validations without improvement; best val "
                      f"l_dir {best['l_dir']:.4f} @ iter {best['it']}", flush=True)
                break
        if (it + 1) % args.save_every == 0:
            save_to(args.out.replace(".pth", f"_iter{it+1}.pth"))

    save_to(args.out.replace(".pth", "_last.pth"))
    v = validate()
    # args.out must carry the BEST model, not the last: the last is the overfit one.
    # eval_servo.py / the laptop deployment both read args.out.
    if best["it"] > 0:
        import shutil
        shutil.copyfile(best_path, args.out)
    else:
        torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                    "iters_done": it + 1}, args.out)
    print(f"[done] {(time.time()-t0)/3600:.2f} h | last val l_dir {v['l_dir']:.4f} "
          f"l_norm {v['l_norm']:.4f} | BEST val l_dir {best['l_dir']:.4f} @ iter "
          f"{best['it']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
