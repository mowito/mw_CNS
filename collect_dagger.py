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
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from scipy.spatial.transform import Rotation as _R

from cns.models.cnsv2_net import build_model
from cns.sim.supervisor import pbvs_straight
from cns.models.hybrid_control import hybrid_velocity, should_use_hybrid, PATCH_PX
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


def expert_action(cur, tar, wP):
    """Paper section G: hybrid (2.5D) control while the image gravity centres are
    far apart -- this is what keeps features in view and is why row (7) of Table I
    (always-PBVS) loses success on large initial viewpoint deviation -- then plain
    PBVS for final precision. NOTE the stored LABEL is always PBVS regardless:
    load_bproc_samples() recomputes it from (poses[i], poses[0], wP), matching
    the paper's "PBVS control (which is directly supervised)".
    """
    gc, gd = _gravity_patch(cur, wP), _gravity_patch(tar, wP)
    if gc is None or gd is None or not should_use_hybrid(gc, gd):
        return pbvs_straight(cur, tar), False
    tcT = np.linalg.inv(tar) @ cur
    Rtc, ttc = tcT[:3, :3], tcT[:3, 3]
    t_c = -Rtc.T @ ttc                        # consistent with PBVS translation term
    th_u = _R.from_matrix(Rtc).as_rotvec()
    return hybrid_velocity(t_c, th_u, gc, gd, Z=1.0, lam=1.0), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True, help="dir for rollout scene_*.npz")
    ap.add_argument("--episodes", type=int, default=50)
    # dt=0.10 / 80 steps: the expert converges in ~53 steps at this dt (measured
    # 100%, TE 1.9mm), and render cost scales with step count -- 0.702s/render on
    # this box, so dt=0.02 (273 steps) would cost 5.3h per 100 episodes.
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="P(expert action) per step. 1=pure expert. See module docstring: "
                         "low beta cannot reach the near-goal states we need.")
    ap.add_argument("--te-thresh", type=float, default=0.002)   # CNS v1 dist_eps
    ap.add_argument("--re-thresh", type=float, default=1.0)     # CNS v1 angle_eps
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_model(K=16, refine_layers=4, ctrl_dim=256).to(dev).eval()
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=dev)
        net.load_state_dict(ck["state_dict"])
        print(f"[dagger] policy {args.ckpt} (iters_done={ck.get('iters_done')})", flush=True)
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
    for ep in range(args.episodes):
        wait_for(os.path.join(W, f"ep{ep}_meta.ok"), args.timeout, f"ep{ep} scene")
        meta = np.load(os.path.join(W, f"ep{ep}_meta.npz"))
        tar, cur, wP, Id = meta["tar"], meta["cur0"], meta["wP"], meta["desired"]
        wPo = wP.mean(0); tcw = np.linalg.inv(tar)
        s = float(np.linalg.norm(wPo @ tcw[:3, :3].T + tcw[:3, 3]))

        images = [Id]           # view 0 = desired, matching bproc_gen's layout
        poses = [tar]
        n_expert = 0
        n_hybrid = 0
        for k in range(args.max_steps):
            req = os.path.join(W, f"ep{ep}_req{k}.npy")
            np.save(req, cur)
            with open(req + ".ok", "w") as f:
                f.write("1")
            resp = os.path.join(W, f"ep{ep}_resp{k}.npy")
            wait_for(resp + ".ok", args.timeout, f"ep{ep} render {k}")
            Ic = np.load(resp)

            # record the VISITED state; the expert label for it is recomputed by
            # load_bproc_samples from (poses[i], poses[0], wP).
            images.append(Ic)
            poses.append(cur.copy())

            if rng.uniform() < args.beta:
                vel, used_hyb = expert_action(cur, tar, wP)
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
        with open(os.path.join(W, f"ep{ep}_end"), "w") as f:
            f.write("1")

        te, re = pose_error(poses[-1], tar)
        err_log.append((te, re))
        np.savez_compressed(os.path.join(args.out, f"scene_{ep:04d}.npz"),
                            images=np.asarray(images, dtype=np.uint8),
                            poses=np.asarray(poses), wP=wP)
        n_saved += 1
        n_pairs += len(images) - 1
        print(f"ep {ep:3d}: {len(images)-1:2d} states  expert {n_expert}/{len(images)-1}  "
              f"hybrid {n_hybrid}  last te {te*1000:7.1f} mm re {re:6.1f} deg", flush=True)

    e = np.asarray(err_log)
    print(f"\nDAGGER_SAVED scenes={n_saved} pairs={n_pairs} -> {args.out}", flush=True)
    print(f"final-state error reached: median TE {np.median(e[:,0])*1000:.1f} mm  "
          f"RE {np.median(e[:,1]):.1f} deg", flush=True)
    print(f"  (low values here mean the rollouts DID reach the near-goal regime, "
          f"which is the whole point of beta>0)", flush=True)


if __name__ == "__main__":
    main()
