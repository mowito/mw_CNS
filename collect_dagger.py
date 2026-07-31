"""DAgger data collection in the BlenderProc (training) domain.

The paper's Fig. 3 pipeline has TWO sampling processes: uniform offline pairs,
and DAgger resampling from rollout states. Only the first was implemented, and
the consequence was measurable -- offline pairs draw desired/current poses
independently, so only ~6% of samples land near the goal, and the policy is at
chance there (translation cosine 0.51 vs 0.90-0.97 far out). Closed loop then
contracts early and stalls or diverges near the goal.

This script rolls trajectories out in Blender and saves every visited state, so
training sees the state distribution a servo actually traverses.

WHY --beta MATTERS (do not default it to 0):
    A pure-policy rollout (beta=0) from a policy that stalls ~1m from the goal
    visits MORE FAR-FIELD states and never reaches the near-goal regime -- it
    would reinforce exactly the distribution we already have. Expert-mixed
    rollouts do converge, and because convergence is geometric (~15%/step at
    dt=0.15) most steps of a converging trajectory are spent near the goal, so
    high beta yields dense near-goal coverage for free.
    Suggested schedule across rounds: beta 1.0 -> 0.7 -> 0.4 -> 0.2, mixing in
    progressively more of the policy's own compounding-error states.

Rollouts are saved in the SAME per-scene .npz layout bproc_gen writes
(images[0]/poses[0] = desired, images[1:]/poses[1:] = visited states, plus wP),
so load_bproc_samples() labels them with the PBVS expert automatically and they
merge into the dataset with merge_scenes.py. No loader changes needed.

    bash scripts/run_dagger_collect.sh checkpoints/cnsv2.pth 50 1.0 data/dagger_r0
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from scipy.spatial.transform import Rotation as _R

from cns.models.cnsv2_net import build_model, build_from_checkpoint
from cns.sim.supervisor import pbvs_straight
from cns.models.hybrid_control import (hybrid_velocity_eq24, should_use_hybrid,
                                       PATCH_PX, LAM)
from cns.utils.perception import in_frame_fraction
from eval_servo import pose_error, integrate
from eval_servo_bproc import wait_for

_FX = _FY = 512.0
_CX = _CY = 256.0
_IMG = 512


def _gravity_patch(wcT, wP):
    """Ground-truth image gravity centre in patch units (expert-side Eq. 25
    surrogate). The EXPERT may use ground truth; the network's own dual-softmax
    gravity centres are what get used at deployment."""
    Rw, t = wcT[:3, :3], wcT[:3, 3]
    pc = (wP - t) @ Rw
    z = pc[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return None
    u = _FX * pc[ok, 0] / z[ok] + _CX
    v = _FY * pc[ok, 1] / z[ok] + _CY
    m = (u >= 0) & (u < _IMG) & (v >= 0) & (v < _IMG)
    if m.sum() == 0:
        return None
    return np.array([u[m].mean(), v[m].mean()]) / PATCH_PX - 0.5


def expert_action(cur, tar, wP, d_star=None, drive="pbvs"):
    """Paper section G: hybrid (2.5D) control while the image gravity centres are
    far apart -- this is what keeps features in view and is why row (7) of Table I
    (always-PBVS) loses success on large initial viewpoint deviation -- then plain
    PBVS for final precision. NOTE the stored LABEL is always PBVS regardless:
    load_bproc_samples() recomputes it from (poses[i], poses[0], wP), matching
    the paper's "PBVS control (which is directly supervised)".

    d_star: scene scale, passed to Eq. 24 as the gravity centre's depth Z. This
    used to be hardcoded Z=1.0, which is only right when d*=1 m (the paper's
    Table I). Our sampler draws r in [0.5,0.9] so d* is 0.53-0.90, and deployment
    passes the real d* (eval_servo_bproc.py). Measured over 271 hybrid steps, that
    mismatch made the collection expert and the deployed controller disagree by a
    MEDIAN 10.3 deg in direction (max 149 deg, cos as low as -0.86) on identical
    states. Labels were unaffected -- they are always PBVS from ground-truth poses
    -- but the visited STATE DISTRIBUTION was the expert's, not the deployed
    controller's, which is the one thing DAgger exists to align.
    """
    # drive="pbvs": plain PBVS every step. The hybrid Eq. 24 path is reachable but
    # OFF by default, because its correctness is not established -- it has already
    # been transcribed wrongly once (Malis Eq. 23 vs paper Eq. 24, see the
    # hybrid_control docstring) and ran with the wrong depth Z for two whole runs.
    # Gate 1 measures plain PBVS at 100% / TE 1.97 mm, so it is the expert we can
    # actually vouch for. This only changes which action DRIVES the rollout; the
    # label was always plain PBVS.
    if drive == "pbvs":
        return pbvs_straight(cur, tar), False
    gc, gd = _gravity_patch(cur, wP), _gravity_patch(tar, wP)
    if gc is None or gd is None or not should_use_hybrid(gc, gd):
        return pbvs_straight(cur, tar), False
    # inv(tar) @ cur maps a point from the CURRENT camera frame to the DESIRED
    # one: dP = R_dc cP + t_dc. That is exactly the (R_dc, t_dc) Malis Eq. 23
    # wants, and theta*u is its rotation vector.
    tcT = np.linalg.inv(tar) @ cur
    R_dc, t_dc = tcT[:3, :3], tcT[:3, 3]
    th_u = _R.from_matrix(R_dc).as_rotvec()
    t_c = -R_dc.T @ t_dc          # desired-cam origin in the current frame
    Z = 1.0 if d_star is None else float(d_star)
    return hybrid_velocity_eq24(t_c, th_u, gc, gd, Z=Z, lam=LAM), True



def _subsample_states(poses, tar, n_keep):
    """Pick visited-state indices that are ~LOG-UNIFORM in translation error.

    A converging rollout is geometric, so its states cluster hard at the goal:
    saving all of them gave a measured median ||vel_si|| of 0.027 (~0.45 of a
    16px patch) with 97.3% below 0.5. Those samples are individually unlearnable
    (paired cos(Fc,Fd) = 0.93 in that bin) and they dominate the sigma_inv
    magnitude loss. Binning by log(te) and taking one state per bin keeps the full
    error range that the closed loop actually traverses, which is the point of
    DAgger, without the pile-up.

    Returns indices into poses (1-based; index 0 is the desired view).
    """
    n = len(poses) - 1
    if n <= n_keep:
        return list(range(1, len(poses)))
    te = np.array([pose_error(poses[i], tar)[0] for i in range(1, len(poses))])
    lo = max(float(te[te > 0].min()) if (te > 0).any() else 1e-4, 1e-4)
    hi = max(float(te.max()), lo * 1.001)
    edges = np.geomspace(lo, hi, n_keep + 1)
    bins = np.clip(np.digitize(te, edges) - 1, 0, n_keep - 1)
    keep = []
    for b in range(n_keep):
        idx = np.where(bins == b)[0]
        if len(idx):
            keep.append(int(idx[len(idx) // 2]) + 1)   # +1: skip the desired view
    return sorted(set(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True, help="dir for rollout scene_*.npz")
    ap.add_argument("--episodes", type=int, default=50)
    # PAPER VALUES. The paper runs dt = 1/50 s and 30 s episodes = 1500 control
    # steps. The old defaults (dt 0.10-0.15, 40-80 steps) were driven purely by
    # BlenderProc's 0.702 s/render, and Sec. 6.3 is blunt about the cost: 40 steps
    # is ~38x less closed-loop integration than the paper, "a far bigger handicap
    # than the patch size". Sub-mm final error comes from geometric decay over
    # ~1500 steps, so a 40-step eval CANNOT reach Table I's TE 0.948 mm no matter
    # how good the policy is. IsaacSim renders ~20x faster, so this is now
    # affordable; episodes still exit early on convergence (~265 steps measured).
    # COLLECTION, not evaluation. The paper's 30 s / 1500 steps is an eval/deploy
    # figure; for data collection the whole converging trajectory is already covered
    # by ~265 steps (measured expert convergence at dt=1/50), and letting an episode
    # that fails the 2mm gate grind out 1500 steps produces ~1200 near-identical
    # near-goal frames and starves episode throughput.
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=1.0 / 50.0)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="P(expert action) per step. 1=pure expert. See module docstring: "
                         "low beta cannot reach the near-goal states we need.")
    ap.add_argument("--te-thresh", type=float, default=0.002)   # CNS v1 dist_eps
    ap.add_argument("--re-thresh", type=float, default=1.0)     # CNS v1 angle_eps
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    # ---- concurrent mode (paper Fig. 3) ---------------------------------
    # In the paper the collector runs BESIDE training and keeps picking up fresh
    # weights, rather than being handed one frozen checkpoint per round. --watch
    # re-reads --ckpt whenever the trainer republishes it (mtime change), so the
    # rollouts track the policy as it improves -- that is what makes this DAgger
    # instead of iterated batch imitation.
    # Keep the visited states LOG-UNIFORM in translation error instead of saving
    # every step. Expert convergence is geometric, so raw rollout states pile up at
    # the goal: a measured round gave median ||vel_si|| 0.027 (~0.45 patch, below
    # what one patch of correspondence can resolve) with 97.3% under 0.5. Feeding
    # that in at --dagger-frac 0.5 would make half of every batch unlearnable and
    # blow up the sigma_inv magnitude target. Log-binning by error reproduces the
    # coverage pose_perturb.py deliberately builds for the offline near-goal set.
    ap.add_argument("--tag", default="",
                    help="filename tag so parallel collectors can share --out "
                         "(Fig. 3 runs Env 1..N); e.g. --tag i1 -> scene_i1_0000.npz")
    ap.add_argument("--states-per-ep", type=int, default=48,
                    help="max saved states per episode, log-uniform in pose error")
    ap.add_argument("--watch", action="store_true",
                    help="reload --ckpt whenever it changes (concurrent DAgger)")
    ap.add_argument("--drive", default="pbvs", choices=["pbvs", "hybrid"],
                    help="which expert action DRIVES the rollout. pbvs (default) is "
                         "the one gate 1 validates at 100%%/1.97mm. hybrid uses the "
                         "Eq. 24 2.5D law, whose implementation is not yet trusted. "
                         "The stored LABEL is plain PBVS either way.")
    # Divergence guard. Without it the only exits are non-finite velocity,
    # convergence, and max_steps -- so a policy-driven rollout that runs away keeps
    # rendering (and SAVING states) until the step budget ends. At beta=0 that is
    # most of the episode budget spent on states hundreds of times further out than
    # anything the uniform half covers.
    ap.add_argument("--max-te-ratio", type=float, default=3.0,
                    help="end the episode once TE exceeds this multiple of the "
                         "INITIAL TE; states after the turn are discarded. 0 disables.")
    ap.add_argument("--max-te", type=float, default=2.0,
                    help="absolute TE bound in metres, applied alongside "
                         "--max-te-ratio. 0 disables.")
    ap.add_argument("--min-in-frame", type=float, default=0.05,
                    help="end the episode once fewer than this fraction of object "
                         "points project inside the image. The TE guard above is "
                         "blind to this: rotating in place leaves TE at 0 while the "
                         "scene leaves the frame (0.00 in-frame at 45 deg of yaw). "
                         "0 disables.")
    ap.add_argument("--beta-iters", type=int, default=0,
                    help="anneal beta over TRAINING iterations (iters_done from the "
                         "synced checkpoint), not episodes. Set this to the trainer's "
                         "--iters. Fig. 3's DAgger share should track training "
                         "progress, and it makes the schedule independent of however "
                         "many episodes a collector happens to get through. "
                         "REQUIRED with --beta-final unless --beta-episodes is given.")
    ap.add_argument("--beta-episodes", type=int, default=0,
                    help="episode-based annealing horizon, if you really want it. "
                         "Do NOT leave this at --episodes: that argument doubles as "
                         "an 'effectively unbounded, killed when training ends' "
                         "sentinel of 100000, so annealing over it is a no-op -- "
                         "beta stayed at 1.00 for every episode of both 40k runs and "
                         "NO DAgger data was ever on-policy (pure behavioural cloning "
                         "on expert rollouts). See the guard below.")
    ap.add_argument("--beta-final", type=float, default=None,
                    help="anneal beta linearly from --beta to this over "
                         "--beta-iters (preferred) or --beta-episodes. "
                         "DAgger wants decreasing expert share; start high because "
                         "an early policy cannot reach the near-goal states.")
    args = ap.parse_args()

    # The bug this guard exists for: --beta-final was silently annealed over
    # --episodes, which callers set to 100000 as an "unbounded" sentinel. beta
    # therefore never left 1.00, so every rollout was 100% expert-driven and the
    # "DAgger" half of the database was plain behavioural cloning on expert
    # trajectories. Nothing in the logs looked wrong -- they just printed
    # "beta 1.00" forever. Refuse to run rather than repeat it.
    if args.beta_final is not None and args.beta_iters <= 0 and args.beta_episodes <= 0:
        raise SystemExit(
            "--beta-final needs an explicit annealing horizon: pass --beta-iters "
            "<trainer --iters> (preferred, tracks training progress) or "
            "--beta-episodes <N>. Annealing over --episodes is what pinned beta at "
            "1.00 for two entire 40k runs.")
    if args.beta_final is not None and args.beta_iters <= 0:
        print(f"[dagger] WARNING: annealing beta over {args.beta_episodes} EPISODES; "
              f"--beta-iters is preferred so the schedule follows training progress",
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = None
    _iters_done = 0
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=dev)
        # Architecture from the checkpoint: fine_dim is an architectural switch.
        net = build_from_checkpoint(ck, dev).eval()
        _iters_done = int(ck.get("iters_done") or 0)
        print(f"[dagger] policy {args.ckpt} (iters_done={_iters_done})", flush=True)
    elif args.beta < 1.0:
        raise SystemExit(f"{args.ckpt} not found and beta<1 needs a policy")

    @torch.no_grad()
    def predict(Ic_np, Id_np, s):
        def t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255.
        raw = net(t(Ic_np), t(Id_np))
        return net.postprocess(raw, torch.tensor([s], device=dev))[0].cpu().numpy()

    W = args.workdir
    wait_for(os.path.join(W, "READY"), args.timeout, "server READY")

    n_saved = n_pairs = 0
    err_log = []
    _ck_mtime = os.path.getmtime(args.ckpt) if os.path.exists(args.ckpt) else 0.0
    _reloads = 0

    def maybe_reload():
        """Pick up freshly published weights. Loads architecture from the file too,
        so a trainer restarted with a different --fine-dim cannot silently mismatch."""
        nonlocal net, _ck_mtime, _reloads, _iters_done
        if not (args.watch and os.path.exists(args.ckpt)):
            return
        m = os.path.getmtime(args.ckpt)
        if m <= _ck_mtime and net is not None:
            return
        for _ in range(5):                     # tolerate a partially-written file
            try:
                ck = torch.load(args.ckpt, map_location=dev)
                net = build_from_checkpoint(ck, dev).eval()
                _ck_mtime, _reloads = m, _reloads + 1
                _iters_done = int(ck.get("iters_done") or 0)
                print(f"[dagger] reloaded policy (iters_done={_iters_done}, "
                      f"reload #{_reloads})", flush=True)
                return
            except Exception as e:
                print(f"[dagger] weight reload retry ({type(e).__name__})", flush=True)
                time.sleep(0.5)

    for ep in range(args.episodes):
        maybe_reload()
        if args.beta_final is None:
            beta = args.beta
        elif args.beta_iters > 0:
            # Track TRAINING progress, so the on-policy share grows with the policy
            # regardless of collector throughput. iters_done comes from the synced
            # checkpoint the collector already reloads.
            frac = min(1.0, max(0.0, _iters_done / float(args.beta_iters)))
            beta = args.beta + (args.beta_final - args.beta) * frac
        else:
            frac = min(1.0, ep / float(max(1, args.beta_episodes - 1)))
            beta = args.beta + (args.beta_final - args.beta) * frac
        wait_for(os.path.join(W, f"ep{ep}_meta.ok"), args.timeout, f"ep{ep} scene")
        meta = np.load(os.path.join(W, f"ep{ep}_meta.npz"))
        tar, cur, wP, Id = meta["tar"], meta["cur0"], meta["wP"], meta["desired"]
        wPo = wP.mean(0); tcw = np.linalg.inv(tar)
        s = float(np.linalg.norm(wPo @ tcw[:3, :3].T + tcw[:3, 3]))

        images = [Id]           # view 0 = desired, matching bproc_gen's layout
        poses = [tar]
        n_expert = 0
        n_hybrid = 0
        te0 = pose_error(cur, tar)[0]     # for the divergence guard
        diverged = False
        lost_view = False
        for k in range(args.max_steps):
            req = os.path.join(W, f"ep{ep}_req{k}.npy")
            np.save(req, cur)
            with open(req + ".ok", "w") as f:
                f.write("1")
            resp = os.path.join(W, f"ep{ep}_resp{k}.npy")
            wait_for(resp + ".ok", args.timeout, f"ep{ep} render {k}")
            Ic = np.load(resp)
            # Delete the per-step IPC files immediately. At ~786KB per render and
            # up to 1500 steps/episode, leaving them on disk is how two collector
            # envs silently wrote 186GB into /tmp and filled the root filesystem,
            # killing the first 40k run at iter 37.8k. The server never re-reads a
            # consumed request or a delivered response, so this is race-free.
            for _p in (resp, resp + ".ok", req, req + ".ok"):
                try:
                    os.remove(_p)
                except OSError:
                    pass

            # record the VISITED state; the expert label for it is recomputed by
            # load_bproc_samples from (poses[i], poses[0], wP).
            images.append(Ic)
            poses.append(cur.copy())

            if rng.uniform() < beta or net is None:
                vel, used_hyb = expert_action(cur, tar, wP, d_star=s,
                                              drive=args.drive)
                n_expert += 1
                n_hybrid += int(used_hyb)
            else:
                vel = predict(Ic, Id, s)
            if not np.all(np.isfinite(vel)):
                break
            cur = integrate(cur, vel, args.dt)
            te, re = pose_error(cur, tar)
            if te < args.te_thresh and re < args.re_thresh:
                break
            # Divergence guard. The check runs on the state AFTER integrating, and a
            # state is only appended at the top of the next iteration, so breaking
            # here means no out-of-bounds state is ever saved -- every kept state has
            # TE <= lim by construction. The pre-turn states of a failed episode are
            # still kept, which is the useful part: they are exactly the on-policy
            # states DAgger is after. Without this the rollout keeps rendering and
            # SAVING until max_steps, filling the pool with states hundreds of times
            # further out than anything the uniform half covers (r in [0.5,0.9] m),
            # frequently with the objects out of frame entirely.
            lim = max(args.max_te_ratio * te0 if args.max_te_ratio > 0 else 0.0,
                      args.max_te if args.max_te > 0 else 0.0)
            if lim > 0 and te > lim:
                diverged = True
                break
            # Out-of-view guard, orthogonal to the TE bound above. Same placement
            # rationale: the check is on the post-integration pose, which is only
            # appended next iteration, so no blank-view state is ever saved.
            if args.min_in_frame > 0 and \
                    in_frame_fraction(cur, wP) < args.min_in_frame:
                lost_view = True
                break
        with open(os.path.join(W, f"ep{ep}_end"), "w") as f:
            f.write("1")

        te, re = pose_error(poses[-1], tar)
        err_log.append((te, re))

        keep = _subsample_states(poses, tar, args.states_per_ep)
        sel_img = [images[0]] + [images[i] for i in keep]
        sel_pose = [poses[0]] + [poses[i] for i in keep]
        # Write to a temp name and rename: the trainer scans this directory
        # concurrently, and a half-written npz reads as a BadZipFile.
        # The temp name MUST already end in .npz -- savez_compressed silently
        # appends .npz otherwise, so ".npz.tmp" became ".npz.tmp.npz" and the
        # rename then failed with FileNotFoundError. The leading dot is what keeps
        # it out of the trainer's scene_*.npz glob.
        stem = f"scene_{args.tag + '_' if args.tag else ''}{ep:04d}"
        tmp = os.path.join(args.out, f".{stem}.tmp.npz")
        np.savez_compressed(tmp,
                            images=np.asarray(sel_img, dtype=np.uint8),
                            poses=np.asarray(sel_pose), wP=wP)
        os.replace(tmp, os.path.join(args.out, f"{stem}.npz"))
        n_saved += 1
        n_pairs += len(sel_img) - 1
        print(f"ep {ep:3d}: {len(images)-1:3d} visited -> {len(sel_img)-1:2d} saved  "
              f"beta {beta:.2f}  expert {n_expert}/{len(images)-1}  hybrid {n_hybrid}  "
              f"last te {te*1000:7.1f} mm re {re:6.1f} deg"
              f"{'  DIVERGED' if diverged else ('  LOST-VIEW' if lost_view else '')}",
              flush=True)

    e = np.asarray(err_log)
    print(f"\nDAGGER_SAVED scenes={n_saved} pairs={n_pairs} -> {args.out}", flush=True)
    print(f"final-state error reached: median TE {np.median(e[:,0])*1000:.1f} mm  "
          f"RE {np.median(e[:,1]):.1f} deg", flush=True)
    print(f"  (low values here mean the rollouts DID reach the near-goal regime, "
          f"which is the whole point of beta>0)", flush=True)


if __name__ == "__main__":
    main()
