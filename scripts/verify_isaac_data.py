"""Gate 2: verify IsaacSim scenes satisfy the renderer contract.

Checks the things that have silently broken before:
  * npz keys / shapes / dtypes match what load_bproc_samples() expects
  * view 0 really is the desired view and views 1..M really differ from it
  * the pose DISTRIBUTION matches cns.sim.pose_sampling, not some older sampler
    (bproc_eval_server once shipped its own copy and produced d* 0.78-2.81 m with
    0.0 deg in-plane roll, versus 0.47-0.93 m / 17 deg -- and nobody noticed for a
    whole DAgger campaign)
  * ||vel_si|| histogram, so we know how much near-goal data exists
  * a contact sheet, because statistics do not catch a black or untextured render

    python3 scripts/verify_isaac_data.py --data data/isaac_smoke
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sheet", default="", help="path for a contact-sheet PNG")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "scene_*.npz")))
    if not files:
        raise SystemExit(f"no scene_*.npz under {args.data}")
    print(f"[verify] {len(files)} scenes in {args.data}")

    z0 = np.load(files[0])
    print(f"[verify] keys: {sorted(z0.files)}")
    for k in ("images", "poses", "wP"):
        if k not in z0.files:
            raise SystemExit(f"[verify] FAIL missing required key '{k}'")
    im, po = z0["images"], z0["poses"]
    print(f"[verify] images {im.shape} {im.dtype}   poses {po.shape} {po.dtype}   "
          f"wP {z0['wP'].shape}")
    assert im.dtype == np.uint8, f"images must be uint8, got {im.dtype}"
    assert im.ndim == 4 and im.shape[-1] == 3, f"images must be [M+1,H,W,3], got {im.shape}"
    assert po.shape[0] == im.shape[0], "poses and images disagree on view count"
    assert po.shape[1:] == (4, 4), f"poses must be 4x4, got {po.shape[1:]}"
    if "K" in z0.files:
        print(f"[verify] K =\n{z0['K']}")
    if "masks" in z0.files:
        m = z0["masks"]
        print(f"[verify] masks {m.shape} {m.dtype}, {len(np.unique(m))} unique ids "
              f"in scene 0")

    # ---- distribution audit over all scenes --------------------------------
    d_des, d_ini, roll_des, roll_ini, phi_des, phi_ini = [], [], [], [], [], []
    flat, dark, ident = 0, 0, 0
    n_obj = []
    for f in files:
        z = np.load(f)
        imgs, poses, wP = z["images"], z["poses"], z["wP"]
        n_obj.append(len(wP))
        centre = wP.mean(0) if len(wP) else np.zeros(3)
        for i, T in enumerate(poses):
            eye = T[:3, 3]
            r = float(np.linalg.norm(eye - centre))
            # elevation above the ground plane, and in-plane roll of camera x-axis
            phi = float(np.degrees(np.arcsin(np.clip((eye[2] - centre[2]) / max(r, 1e-9), -1, 1))))
            # camera x axis projected on world xy -> roll about the viewing axis
            xax = T[:3, 0]
            roll = float(np.degrees(np.arctan2(xax[2], np.linalg.norm(xax[:2]) + 1e-12)))
            (d_des if i == 0 else d_ini).append(r)
            (phi_des if i == 0 else phi_ini).append(phi)
            (roll_des if i == 0 else roll_ini).append(abs(roll))
        for i in range(imgs.shape[0]):
            if imgs[i].std() < 1.0:
                flat += 1
            if imgs[i].mean() < 5.0:
                dark += 1
        for i in range(1, imgs.shape[0]):
            if np.abs(imgs[i].astype(np.int16) - imgs[0].astype(np.int16)).mean() < 1.0:
                ident += 1

    def stat(name, v, expect):
        v = np.asarray(v)
        print(f"  {name:<26} min {v.min():7.3f}  max {v.max():7.3f}  "
              f"mean {v.mean():7.3f}   expect {expect}")

    print("\n[verify] pose distribution (vs cns/sim/pose_sampling.py):")
    stat("desired  radius (m)", d_des, "0.50-0.90")
    stat("desired  elevation (deg)", phi_des, "70-90")
    stat("desired  |roll| (deg)", roll_des, "<=15")
    stat("initial  radius (m)", d_ini, "0.50-0.90")
    stat("initial  elevation (deg)", phi_ini, "30-90")
    stat("initial  |roll| (deg)", roll_ini, "<=60, NOT 0")
    print(f"  objects per scene          min {min(n_obj)} max {max(n_obj)} "
          f"mean {np.mean(n_obj):.2f}   expect 1-6")

    # Distinguish "a few bad frames" from "systemically wrong". A handful of dark
    # renders is label noise worth reporting; aborting an 85-minute training run
    # over 0.25% of frames would be the wrong call. A zero roll spread, by contrast,
    # means the wrong pose sampler is wired in and every sample is affected.
    n_frames = sum(np.load(f)["images"].shape[0] for f in files)
    bad_frac = (flat + dark + ident) / max(n_frames, 1)
    print(f"\n[verify] image sanity: {flat} flat (std<1), {dark} near-black, "
          f"{ident} current views identical to their desired view "
          f"({100 * bad_frac:.2f}% of {n_frames} frames)")
    TOL = 0.01
    bad = False
    if bad_frac > TOL:
        print(f"[verify] FAIL {100*bad_frac:.2f}% unusable frames exceeds the "
              f"{100*TOL:.0f}% tolerance")
        bad = True
    elif flat or dark or ident:
        print(f"[verify] (tolerated: under the {100*TOL:.0f}% threshold)")
    if max(roll_ini) < 1.0:
        print("[verify] FAIL in-plane roll is ~0 for initial poses -- this is the "
              "exact symptom of the duplicated old sampler (gotcha 6.3b)")
        bad = True

    # ---- labels through the real loader ------------------------------------
    try:
        import torch
        from cns.sim.cnsv2_data import load_bproc_samples
    except ImportError as e:
        print(f"\n[verify] skipping label check ({e}); run under the cnsv2 env")
        return 1 if bad else 0

    # Contract check on a SUBSET: load_bproc_samples materializes Ic AND Id for
    # every pair, so on 2500 scenes (15k pairs) that is ~23.6GB of uint8 just to
    # read labels. A few scenes prove the loader still works.
    import tempfile, shutil as _sh
    sub = tempfile.mkdtemp(prefix="verify_sub_")
    try:
        for f in files[:8]:
            _sh.copy2(f, sub)
        s = load_bproc_samples(sub)
        print(f"\n[verify] load_bproc_samples on {min(8, len(files))} scenes: "
              f"{s['Ic'].shape[0]} pairs, Ic {tuple(s['Ic'].shape)} {s['Ic'].dtype}")
    finally:
        _sh.rmtree(sub, ignore_errors=True)

    # Label distribution over ALL scenes, computed WITHOUT touching images: npz is
    # lazy, so reading poses/wP does not decompress the image array. Same trick
    # build_feature_cache's pass 1 uses.
    from cns.sim.supervisor import supervisor_vel, Policy
    from cns.utils.perception import CameraIntrinsic
    intr = CameraIntrinsic(512, 512, 512, 512, 256, 256)
    dummy, dz = np.zeros((1, 2)), np.ones(1)
    vs, tp = [], []
    for f in files:
        z = np.load(f)
        poses, wP = z["poses"], z["wP"]
        for i in range(1, poses.shape[0]):
            _, (tpo, vsi) = supervisor_vel(Policy.PBVS_Straight, dummy, dz, dummy,
                                           dz, intr, poses[i], poses[0], wP)
            vs.append(vsi); tp.append(tpo)
    s = {"vel_si": torch.tensor(np.stack(vs), dtype=torch.float32),
         "tPo_norm": torch.tensor(np.array(tp), dtype=torch.float32)}
    v = s["vel_si"].norm(dim=-1)
    print(f"[verify] labelled {len(v)} pairs from all {len(files)} scenes "
          f"(images not loaded)")
    print(f"[verify] ||vel_si||  min {v.min():.4f}  median {v.median():.4f}  "
          f"max {v.max():.4f}")
    edges = [0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9]
    print("[verify] ||vel_si|| histogram:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((v >= lo) & (v < hi)).sum())
        lbl = f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},inf)"
        print(f"    {lbl:>14}  {n:6d}  {100*n/len(v):5.1f}%")
    print(f"[verify] tPo_norm (d*)  min {s['tPo_norm'].min():.3f}  "
          f"max {s['tPo_norm'].max():.3f}")

    if args.sheet:
        try:
            import imageio.v3 as iio
            rows = []
            for f in files[:4]:
                z = np.load(f)
                rows.append(np.concatenate(list(z["images"][:5]), axis=1))
            iio.imwrite(args.sheet, np.concatenate(rows, axis=0))
            print(f"[verify] contact sheet -> {args.sheet} "
                  f"(col 0 = desired, cols 1-4 = currents)")
        except Exception as e:
            print(f"[verify] contact sheet failed: {e}")

    print("\n[verify] " + ("FAILED" if bad else "PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
