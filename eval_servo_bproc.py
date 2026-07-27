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

from cns.models.cnsv2_net import build_model
from eval_servo import pose_error, integrate      # identical metrics + integrator


def wait_for(path, timeout, what):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out after {timeout}s waiting for {what} ({path})")
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--te-thresh", type=float, default=0.005)   # 5 mm
    ap.add_argument("--re-thresh", type=float, default=1.0)     # 1 deg
    ap.add_argument("--oracle", action="store_true",
                    help="ground-truth PBVS instead of the model. NOTE: this never "
                         "looks at an image, so it validates the loop/integrator "
                         "ONLY -- it cannot detect an image-domain problem.")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    W = args.workdir
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_model(K=16, refine_layers=4, ctrl_dim=256).to(dev).eval()
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=dev)
        net.load_state_dict(ck["state_dict"])
        print(f"loaded {args.ckpt} (iters_done={ck.get('iters_done')})", flush=True)
    elif not args.oracle:
        raise SystemExit(f"{args.ckpt} not found")

    @torch.no_grad()
    def predict(Ic_np, Id_np, s):
        def t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev) / 255.
        raw = net(t(Ic_np), t(Id_np))
        return net.postprocess(raw, torch.tensor([s], device=dev))[0].cpu().numpy()

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
                vel = predict(Ic, Id, s)
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
              f"re {re0:6.1f}->{reN:6.1f} deg | {len(traj)-1:2d} steps | "
              f"{'OK' if ok else '--'}", flush=True)

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
