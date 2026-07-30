"""Verification gates 1, 3, 4 from PAPER_FAITHFUL_5090.md Sec. 7.

Run these BEFORE any long training run. Each one has caught a real bug in this
project's history.

  Gate 1  Expert sanity. Pure PBVS at the paper's dt=1/50 must converge on the
          CNS-v1 sampling distribution. Pose-space only, no rendering.
  Gate 3  Input ablation. Zero/shuffle each controller input stream and confirm
          the output moves. The probability grid P once contributed 2.3e-07 to the
          output -- probabilistic correspondence IS the method, so if P does not
          measurably matter, the model is not the paper's model.
  Gate 4  Overfit. A healthy head reaches l_dir ~1e-4 on N=4 and should fit N>=32
          with LOW target pairwise cosine. A single sample proves nothing: a
          constant predictor scores ~1.0 on it. Always compare against the
          BEST-CONSTANT baseline, which is 1 - cos(mean direction), not 1.0.

    python scripts/run_gates.py --gates 1,3,4 --data data/isaac_smoke
"""
import argparse, glob, os, sys, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------- gate 1
def gate1_expert(n_episodes=20, dt=1.0 / 50.0, max_steps=1500):
    """Pure PBVS from CNS-v1-sampled initial poses must reach the goal."""
    from scipy.spatial.transform import Rotation as R
    from cns.sim.pose_sampling import sample_desired, sample_initial, DIST_EPS, ANGLE_EPS
    from cns.sim.supervisor import pbvs_straight

    def pose_error(cur, tar):
        dT = np.linalg.inv(tar) @ cur
        te = float(np.linalg.norm(dT[:3, 3]))
        re = float(np.degrees(np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec())))
        return te, re

    def integrate(wcT, vel, dt):
        dR = R.from_rotvec(vel[3:] * dt).as_matrix()
        dT = np.eye(4); dT[:3, :3] = dR; dT[:3, 3] = vel[:3] * dt
        return wcT @ dT

    rng = np.random.RandomState(0)
    ok, tes, res, steps = 0, [], [], []
    for ep in range(n_episodes):
        centre = np.zeros(3)
        tar = sample_desired(rng, centre)
        cur = sample_initial(rng, centre)
        te0, re0 = pose_error(cur, tar)
        for k in range(max_steps):
            te, re = pose_error(cur, tar)
            if te < DIST_EPS and re < ANGLE_EPS:
                break
            cur = integrate(cur, pbvs_straight(cur, tar), dt)
        te, re = pose_error(cur, tar)
        good = te < DIST_EPS and re < ANGLE_EPS
        ok += bool(good)
        tes.append(te * 1000); res.append(re); steps.append(k + 1)
    print(f"  SR {ok}/{n_episodes}   TE {np.mean(tes):.2f}+-{np.std(tes):.2f} mm   "
          f"RE {np.mean(res):.3f}+-{np.std(res):.3f} deg   steps {np.mean(steps):.0f}")
    print(f"  expected (measured previously): 100%, TE ~1.97mm, RE ~0.37deg, ~273 steps")
    passed = ok == n_episodes
    print(f"  GATE 1: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------- gate 3
def gate3_ablation(data, device="cuda", B=8):
    """Zero / shuffle each controller input; a stream that changes nothing is dead.

    Uses REAL rendered pairs, not random tensors. With random features the score
    matrix is near-uniform for every sample, so P barely differs between samples
    and 'shuffle P across the batch' looks like a no-op even when the wiring is
    correct. Only real images make the shuffle test meaningful.

    NOTE this runs on an UNTRAINED net, so it deliberately does NOT check output
    diversity: at initialization the action token produces nearly the same vector
    for any input (measured pairwise cosine ~0.996), which is expected. The
    constant-predictor collapse that Sec. 6.1/6.2 warns about is a TRAINED-model
    failure and is checked in gate 4 against the best-constant baseline.
    """
    import torch
    from cns.models.cnsv2_net import build_model
    from cns.sim.cnsv2_data import load_bproc_samples

    torch.manual_seed(0)
    K = 16
    net = build_model(K=K, refine_layers=4, ctrl_dim=256, fine_dim=128).to(device).eval()

    s = load_bproc_samples(data)
    B = min(B, s["Ic"].shape[0])
    Ic = s["Ic"][:B].to(device).float().div_(255.0)
    Id = s["Id"][:B].to(device).float().div_(255.0)
    with torch.no_grad():
        Fc, Fd = net.backbone_features(Ic, Id)

    with torch.no_grad():
        rfc, rfd = net.refine(Fc, Fd)
        out = net.prob(rfc, rfd)
        P, fine = out["P"], net.fine(Ic, Id)
        base, _, _ = net.controller(rfc, P, fine=fine)

        def rel(v):
            return float((v - base).norm() / base.norm().clamp_min(1e-12))

        z_P, _, _ = net.controller(rfc, torch.zeros_like(P), fine=fine)
        s_P, _, _ = net.controller(rfc, P[torch.randperm(B, device=device)], fine=fine)
        z_f, _, _ = net.controller(rfc, P, fine=torch.zeros_like(fine))
        s_f, _, _ = net.controller(rfc, P, fine=fine[torch.randperm(B, device=device)])
        z_c, _, _ = net.controller(torch.zeros_like(rfc), P, fine=fine)

        rows = [("P zeroed", rel(z_P)), ("P shuffled", rel(s_P)),
                ("fine CNN zeroed", rel(z_f)), ("fine CNN shuffled", rel(s_f)),
                ("coarse feats zeroed", rel(z_c))]
        d = base / base.norm(dim=-1, keepdim=True)
        off = (d @ d.T)[~torch.eye(B, dtype=bool, device=device)]

    print(f"  {'ablation':<24} {'relative output change':>22}")
    for name, v in rows:
        flag = "" if v > 1e-3 else "   <-- DEAD INPUT"
        print(f"  {name:<24} {v:>22.6f}{flag}")
    print(f"  (untrained-net output pairwise cosine {off.mean():.4f} -- expected "
          f"near 1 at init, not a failure; gate 4 tests collapse)")
    passed = all(v > 1e-3 for _, v in rows)
    print(f"  GATE 3: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------- gate 4
def gate4_overfit(data, n=32, iters=1500, lr=5e-5, device="cuda"):
    """Fit a small batch of REAL samples. Compare against the best-constant baseline."""
    import torch
    from cns.models.cnsv2_net import build_model
    from cns.sim.cnsv2_data import load_bproc_samples

    s = load_bproc_samples(data)
    N = min(n, s["vel_si"].shape[0])
    vel = s["vel_si"][:N].to(device)

    # Best-constant baseline: a model that learned NOTHING scores this, not 1.0.
    d = vel / vel.norm(dim=-1, keepdim=True)
    mean_dir = d.mean(0, keepdim=True)
    mean_dir = mean_dir / mean_dir.norm()
    const_l_dir = float((1 - (d * mean_dir).sum(-1)).mean())
    pw = (d @ d.T)
    off = pw[~torch.eye(N, dtype=bool, device=device)]
    print(f"  N={N}  target pairwise cosine mean {off.mean():.4f} "
          f"(low is what makes this test meaningful)")
    print(f"  BEST-CONSTANT baseline l_dir = {const_l_dir:.4f}  <- beat this, not 1.0")

    net = build_model(K=16, refine_layers=4, ctrl_dim=256, fine_dim=128).to(device)
    Ic = s["Ic"][:N].to(device).float().div_(255.0)
    Id = s["Id"][:N].to(device).float().div_(255.0)
    with torch.no_grad():
        Fc, Fd = net.backbone_features(Ic, Id)

    decay, no_decay = net.get_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    net.train()
    t0 = time.time()
    best = float("inf")
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = net.head_forward(Fc, Fd, Ic=Ic, Id=Id)
        res, loss = net.objectives(raw, vel, 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        best = min(best, res["l_dir"])
        if (it + 1) % 250 == 0:
            print(f"    iter {it+1:5d}  l_dir {res['l_dir']:.5f}  "
                  f"l_norm {res['l_norm']:.4f}  best {best:.5f}  "
                  f"({(it+1)/(time.time()-t0):.1f} it/s)")
    # Must beat the constant baseline by a wide margin to prove it learned structure.
    passed = best < 0.25 * const_l_dir
    print(f"  best l_dir {best:.5f} vs constant {const_l_dir:.4f} "
          f"(ratio {best/const_l_dir:.3f})")
    print(f"  GATE 4: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="1,3,4")
    ap.add_argument("--data", default="data/isaac_smoke")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--overfit-n", type=int, default=32)
    ap.add_argument("--overfit-iters", type=int, default=1500)
    args = ap.parse_args()
    want = {g.strip() for g in args.gates.split(",")}
    results = {}

    if "1" in want:
        print("\n=== GATE 1: expert sanity (pure PBVS, dt=1/50) ===")
        results["1"] = gate1_expert(n_episodes=args.episodes)
    if "3" in want:
        print("\n=== GATE 3: controller input ablation ===")
        results["3"] = gate3_ablation(args.data)
    if "4" in want:
        print(f"\n=== GATE 4: overfit N={args.overfit_n} on real renders ===")
        results["4"] = gate4_overfit(args.data, n=args.overfit_n,
                                     iters=args.overfit_iters)

    print("\n=== SUMMARY ===")
    for g in sorted(results):
        print(f"  gate {g}: {'PASS' if results[g] else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
