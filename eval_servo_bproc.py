"""Closed-loop servo evaluation IN THE BLENDERPROC DOMAIN (the training domain).

`eval_servo.py` closes the loop with PyBullet, but the policy is trained on
BlenderProc photorealistic renders. Measured on the same checkpoint: held-out
Blender l_dir 0.129 vs live PyBullet 0.852 -- worse than the 0.721 best-constant
on those pairs, with frozen-RADIO feature cosine only 0.36 across renderers. So
a PyBullet SR says nothing about the policy. This script instead drives a
persistent BlenderProc render server (cns/render/bproc_eval_server.py) so every
observation comes from the same distribution the model was trained on.

Launch the server first (it prints READY into --workdir), then:

    python3 eval_servo_bproc.py --ckpt checkpoints/cnsv2.pth \
        --workdir /tmp/servo_ipc --episodes 20

Or use scripts/run_bproc_eval.sh which starts both and cleans up.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from cns.models.cnsv2_net import build_model, build_from_checkpoint
from cns.models.controller import sigma
from cns.models.denorm import pbvs_inverse
from cns.models.hybrid_control import hybrid_velocity_eq24, should_use_hybrid
from eval_servo import pose_error, integrate      # identical metrics + integrator


def wait_for(path, timeout, what, poll=0.002):
    """poll defaults to 2ms, not 20ms: a 245-step episode does ~250 round trips, so
    a 20ms granularity adds ~10ms of pure sleep to each one."""
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out after {timeout}s waiting for {what} ({path})")
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    # PAPER VALUES. The paper runs dt = 1/50 s and 30 s episodes = 1500 control
    # steps. The old defaults (dt 0.10-0.15, 40-80 steps) were driven purely by
    # BlenderProc's 0.702 s/render, and Sec. 6.3 is blunt about the cost: 40 steps
    # is ~38x less closed-loop integration than the paper, "a far bigger handicap
    # than the patch size". Sub-mm final error comes from geometric decay over
    # ~1500 steps, so a 40-step eval CANNOT reach Table I's TE 0.948 mm no matter
    # how good the policy is. IsaacSim renders ~20x faster, so this is now
    # affordable; episodes still exit early on convergence (~265 steps measured).
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=1.0 / 50.0)
    ap.add_argument("--te-thresh", type=float, default=0.005)   # 5 mm
    ap.add_argument("--re-thresh", type=float, default=1.0)     # 1 deg
    ap.add_argument("--oracle", action="store_true",
                    help="ground-truth PBVS instead of the model. NOTE: this never "
                         "looks at an image, so it validates the loop/integrator "
                         "ONLY -- it cannot detect an image-domain problem.")
    ap.add_argument("--deploy", default="hybrid", choices=["hybrid", "pbvs"],
                    help="hybrid = Table I row 1 (Sec. G switch, the paper's "
                         "deployment); pbvs = row 7 ablation (raw policy always)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    W = args.workdir
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=dev)
        # Architecture from the checkpoint: fine_dim is an architectural switch.
        net = build_from_checkpoint(ck, dev).eval()
        print(f"loaded {args.ckpt} (iters_done={ck.get('iters_done')})", flush=True)
    elif not args.oracle:
        raise SystemExit(f"{args.ckpt} not found")
    else:
        net = build_model(K=16, refine_layers=4, ctrl_dim=256).to(dev).eval()

    @torch.no_grad()
    def predict(Ic_np, Id_np, s):
        """Paper Table I row 1 deployment (--deploy hybrid, the default):

        while ||cX_g - dX_g|| > 0.1*sqrt(N16):   Sec. G safe velocity control
            v = -lam J~^-1 e   built from NETWORK estimates -- gravity centres
            from the dual-softmax scores (Eq. 25) and {R^, t^} recovered from the
            predicted velocity via PBVS^-1 (Eq. 18)
        else:                                      raw policy output (imitates PBVS,
            "which is directly supervised")

        --deploy pbvs is Table I row (7), the ablation the paper reports dropping
        to 18/20 with failures at large initial viewpoint deviation. Our offline
        probe shows the same signature (far-field [2,inf) bin: dir cos 0.77 and
        the largest bias, 0.148), so the hybrid stage is doing real work here, not
        ceremony. Returns (vel_6d_real_units, mode_str).
        """
        def t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255.
        Ic_t, Id_t = t(Ic_np), t(Id_np)
        rfc, rfd = net.features(Ic_t, Id_t)
        out = net.prob(rfc, rfd)
        fine = net.fine(Ic_t, Id_t) if net.fine is not None else None
        vec, log_norm, _ = net.controller(rfc, out["P"], fine=fine)
        raw = (vec, log_norm, None)

        if args.deploy == "hybrid":
            cXg, dXg, _ = net.prob.gravity_centers(rfc, rfd)
            cXg = cXg[0].cpu().numpy()
            dXg = dXg[0].cpu().numpy()
            if should_use_hybrid(cXg, dXg):
                # normalized velocity v~ (unit world), then Eq. 18: {R^, t^} <- PBVS^-1
                d = vec[0] / vec[0].norm().clamp_min(1e-9)
                v_norm = (d * sigma(log_norm[0])).cpu().numpy()
                R_dc, t_dc = pbvs_inverse(v_norm)
                th_u = -v_norm[3:]                 # Eq. 5: omega = -lam*theta*u, lam=1
                # position of the desired camera origin in the CURRENT frame; t^ is
                # in the unit world (Z~_d = 1) so the real scene scale s applies.
                t_c = -(R_dc.T @ t_dc) * s
                return hybrid_velocity_eq24(t_c, th_u, cXg, dXg,
                                            Z=s, lam=1.0), "hyb"

        return (net.postprocess(raw, torch.tensor([s], device=dev))[0].cpu().numpy(),
                "pbvs")

    print(f"[client] waiting for server READY in {W}", flush=True)
    wait_for(os.path.join(W, "READY"), args.timeout, "server READY")

    rows = []
    for ep in range(args.episodes):
        wait_for(os.path.join(W, f"ep{ep}_meta.ok"), args.timeout, f"ep{ep} scene")
        meta = np.load(os.path.join(W, f"ep{ep}_meta.npz"))
        tar, cur, wP, Id = meta["tar"], meta["cur0"], meta["wP"], meta["desired"]

        # scene scale d* = tPo_norm at the desired pose -- computed exactly as
        # supervisor_vel does when labelling training data.
        wPo = wP.mean(0); tcw = np.linalg.inv(tar)
        s = float(np.linalg.norm(wPo @ tcw[:3, :3].T + tcw[:3, 3]))

        te0, re0 = pose_error(cur, tar)
        traj = [(te0, re0)]
        n_hyb = 0
        for k in range(args.max_steps):
            if args.oracle:
                from cns.sim.supervisor import pbvs_straight
                vel = pbvs_straight(cur, tar)
            else:
                req = os.path.join(W, f"ep{ep}_req{k}.npy")
                np.save(req, cur)
                with open(req + ".ok", "w") as f:
                    f.write("1")
                resp = os.path.join(W, f"ep{ep}_resp{k}.npy")
                wait_for(resp + ".ok", args.timeout, f"ep{ep} render {k}")
                Ic = np.load(resp)
                # Delete the per-step IPC files immediately. At ~786KB per render
                # and up to 1500 steps/episode, leaving them on disk is how two
                # collector envs silently wrote 186GB into /tmp and filled the
                # root filesystem, killing the first 40k run at iter 37.8k. The
                # server never re-reads a consumed request or a delivered
                # response, so this is race-free.
                for _p in (resp, resp + ".ok", req, req + ".ok"):
                    try:
                        os.remove(_p)
                    except OSError:
                        pass
                vel, mode = predict(Ic, Id, s)
                n_hyb += (mode == "hyb")
            if not np.all(np.isfinite(vel)):
                print(f"  ep {ep}: non-finite velocity at step {k}", flush=True)
                break
            cur = integrate(cur, vel, args.dt)
            traj.append(pose_error(cur, tar))
            if traj[-1][0] < args.te_thresh and traj[-1][1] < args.re_thresh:
                break                                    # converged, stop early
        with open(os.path.join(W, f"ep{ep}_end"), "w") as f:
            f.write("1")

        teN, reN = traj[-1]
        ok = teN < args.te_thresh and reN < args.re_thresh
        rows.append((te0, re0, teN, reN, float(ok)))
        print(f"ep {ep:2d}: te {te0*1000:7.1f}->{teN*1000:7.1f} mm | "
              f"re {re0:6.1f}->{reN:6.1f} deg | {len(traj)-1:4d} steps | "
              f"hyb {n_hyb:4d} | {'OK' if ok else '--'}", flush=True)

    rows = np.asarray(rows, float)
    sr = rows[:, 4].mean()
    succ = rows[rows[:, 4] == 1]
    print("\n==== SERVO EVAL (BlenderProc domain) ====")
    print(f"episodes: {len(rows)}  |  SR: {sr*100:.0f}%")
    if len(succ):
        print(f"successful: median TE {np.median(succ[:,2])*1000:.2f} mm, "
              f"RE {np.median(succ[:,3]):.3f} deg")
    print(f"all: median final TE {np.median(rows[:,2])*1000:.1f} mm, "
          f"RE {np.median(rows[:,3]):.1f} deg")
    print(f"(init error median: TE {np.median(rows[:,0])*1000:.0f} mm, "
          f"RE {np.median(rows[:,1]):.0f} deg)")
    # error-reduction ratio: <1 means it moved toward the goal even without
    # hitting the 5mm/1deg gate, which SR alone hides.
    print(f"median TE ratio final/init = {np.median(rows[:,2]/np.maximum(rows[:,0],1e-9)):.3f}"
          f"   RE ratio = {np.median(rows[:,3]/np.maximum(rows[:,1],1e-9)):.3f}")


if __name__ == "__main__":
    main()
