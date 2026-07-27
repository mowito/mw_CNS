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
from cns.sim.cnsv2_data import load_bproc_samples, precompute_backbone_features

CONFIG = {"K": 16, "feat_dim": 768, "intrinsic": {"fx": 512, "fy": 512, "cx": 256,
          "cy": 256, "H1": 512, "W1": 512}, "d_star": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="BlenderProc scene dir (scene_*.npz)")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
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
    # This head overfits 1850 pairs by ~iter 1000 (val l_dir bottoms at 0.596 then
    # climbs back above the best-constant baseline by iter 2000 while train keeps
    # falling). Without best-tracking the FINAL checkpoint is the worst one, so
    # track the best val l_dir and stop when it stops improving.
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many validations with no val l_dir gain (0=off)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[load] {args.data}", flush=True)
    net = build_model(K=CONFIG["K"], refine_layers=4, ctrl_dim=256).to(dev)

    # Cache precomputed frozen-ViT features to disk so job restarts are cheap
    # (skip render-load + ~5min backbone recompute on every resume).
    cache = args.out.replace(".pth", "_feat.pt")
    import glob as _g
    n_scenes = len(_g.glob(os.path.join(args.data, "scene_*.npz")))
    if os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu")
        Fc, Fd, vsi, tr, va = blob["Fc"], blob["Fd"], blob["vsi"], blob["tr"], blob["va"]
        # The cache used to be trusted on existence alone, so rendering MORE scenes
        # into --data and relaunching would silently keep training on the old
        # sample set. Refuse to run when the scene count no longer matches.
        cached_scenes = blob.get("n_scenes")
        if cached_scenes is None:
            print(f"[feat] WARNING: cache {cache} predates the scene-count guard; "
                  f"it holds {Fc.shape[0]} samples while {args.data} now has "
                  f"{n_scenes} scenes. Delete the cache if you added scenes.", flush=True)
        elif cached_scenes != n_scenes:
            raise SystemExit(
                f"[feat] STALE CACHE: {cache} was built from {cached_scenes} scenes but "
                f"{args.data} now has {n_scenes}. Training on it would ignore the new "
                f"scenes. Delete it and rerun:\n    rm {cache}")
        print(f"[feat] loaded cache {cache}: {Fc.shape[0]} samples "
              f"({cached_scenes if cached_scenes is not None else '?'} scenes)", flush=True)
    else:
        d = load_bproc_samples(args.data)
        n = d["Ic"].shape[0]
        nv = max(8, int(n * args.val_frac))
        perm = torch.randperm(n)
        tr, va = perm[nv:], perm[:nv]
        print(f"[load] {n} samples -> train {len(tr)} / val {len(va)}; precomputing feats...", flush=True)
        import gc
        Fc = precompute_backbone_features(net, d["Ic"], dev)   # CPU fp16
        d["Ic"] = None; gc.collect()      # release before allocating Fd
        Fd = precompute_backbone_features(net, d["Id"], dev)
        d["Id"] = None; gc.collect()
        vsi = d["vel_si"]
        torch.save({"Fc": Fc, "Fd": Fd, "vsi": vsi, "tr": tr, "va": va,
                    "n_scenes": n_scenes}, cache)
        del d                                    # free ~11GB of raw images (avoid OOM)
        import gc; gc.collect()
        print(f"[feat] cached -> {cache}", flush=True)

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

    def batch_forward(idx):
        fc = Fc[idx].to(dev).float(); fd = Fd[idx].to(dev).float()
        return net.head_forward(fc, fd)

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
            res, _ = net.objectives(batch_forward(chunk), vsi[chunk].to(dev))
            for k, v in res.items():
                acc[k] = acc.get(k, 0.0) + v * len(chunk)
            seen += len(chunk)
        net.train()
        return {k: v / seen for k, v in acc.items()}

    def save_to(path):
        torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "iters_done": it + 1}, path)

    best_path = args.out.replace(".pth", "_best.pth")
    best = {"l_dir": float("inf"), "it": -1}
    stale = 0
    print(f"[train] {args.iters} iters (from {start_it}), batch {args.batch}, fp32", flush=True)
    net.train(); t0 = time.time(); hist = []
    for it in range(start_it, args.iters):
        idx = tr[torch.randint(0, len(tr), (args.batch,))]
        opt.zero_grad(set_to_none=True)
        raw = batch_forward(idx)
        _, loss = net.objectives(raw, vsi[idx].to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step(); sched.step()
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
