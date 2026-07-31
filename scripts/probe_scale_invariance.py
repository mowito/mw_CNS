"""Does a cube of size S at distance D project to the same pixels as size kS at kD?

This is the assumption the scale-invariant label rests on (supervisor.py:290 divides
translation by d*), and it is worth checking in the REAL renderer rather than trusting
the ideal pinhole algebra. Tested with a plain cube so the measurement is a silhouette
area, not an appearance judgement.

  ratio-matched   S/D equal   -> images must be IDENTICAL (this is the claim)
  controls        S/D differs -> images must DIFFER (proves the test can detect change)

The silhouette is measured by differencing against a cube-hidden render of the same
pose, so nothing depends on a colour threshold. The ground plane is hidden and the
dome carries no HDRI, giving a uniform background: the ground is 6 m of fixed
geometry that does NOT scale with the cube, so leaving it in would correctly break
the invariance and obscure what is being tested.

    ~/isaacsim/python.sh scripts/probe_scale_invariance.py --out /tmp/scale.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# S/D = 0.2 for the matched set; the controls change S/D by exactly 3x either way.
CASES = [
    ("A_S0.10_D0.50", 0.10, 0.50),      # matched, S/D = 0.200
    ("B_S0.30_D1.50", 0.30, 1.50),      # matched, S/D = 0.200
    ("C_S0.60_D3.00", 0.60, 3.00),      # matched, S/D = 0.200
    ("D_ctrl_S0.10_D1.50", 0.10, 1.50),  # control, S/D = 0.067 -> must look SMALLER
    ("E_ctrl_S0.30_D0.50", 0.30, 0.50),  # control, S/D = 0.600 -> must look BIGGER
]
DIR = np.array([0.4, -0.8, 0.45])       # fixed view direction, deliberately not axis-aligned


def look_at_cv(eye, center, up=(0, 0, 1)):
    eye = np.asarray(eye, float); center = np.asarray(center, float)
    z = center - eye; z /= (np.linalg.norm(z) + 1e-9)
    up = np.asarray(up, float)
    if abs(np.dot(z, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], axis=1); T[:3, 3] = eye
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/scale_invariance.npz")
    ap.add_argument("--usd", default="data/gso_usd")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    from cns.render.isaac_scene import launch
    app = launch(headless=True, gpu=args.gpu)

    from cns.render.isaac_scene import IsaacSceneGen
    from pxr import UsdGeom, Gf

    # No hdri/tex: a bare dome gives a uniform background. reset() is never called,
    # so no GSO objects are ever added.
    gen = IsaacSceneGen(args.usd, None, None, rt_subframes=8, want_masks=False)
    UsdGeom.Imageable(gen.stage.GetPrimAtPath("/World/ground")).MakeInvisible()
    gen.dome.GetIntensityAttr().Set(1500.0)

    cube = UsdGeom.Cube.Define(gen.stage, "/World/cube")   # size 1 -> spans -0.5..0.5
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    scale_op = xf.AddScaleOp()
    img = UsdGeom.Imageable(cube.GetPrim())

    d = DIR / np.linalg.norm(DIR)
    gen.warmup(16)

    shots, rows = {}, []
    for name, S, D in CASES:
        scale_op.Set(Gf.Vec3f(S, S, S))
        pose = look_at_cv(d * D, np.zeros(3))

        img.MakeInvisible()                       # background at this exact pose
        gen.warmup(4)
        bg, _ = gen.render(pose, want_mask=False)
        img.MakeVisible()
        gen.warmup(4)
        im, _ = gen.render(pose, want_mask=False)

        sil = np.abs(im.astype(np.int16) - bg.astype(np.int16)).max(-1) > 8
        n = int(sil.sum())
        if n:
            ys, xs = np.nonzero(sil)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            w, h = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            bbox, w, h, cx, cy = (0, 0, 0, 0), 0, 0, float("nan"), float("nan")
        shots[name] = im
        rows.append((name, S, D, S / D, n, w, h, cx, cy))
        print(f"[scale] {name:<20} S/D={S/D:.3f}  silhouette={n:7d}px  "
              f"bbox={w}x{h}  centroid=({cx:.1f},{cy:.1f})", flush=True)

    print("\n[scale] pairwise mean|A-B| over FULL images (matched set A,B,C):")
    keys = [r[0] for r in rows[:3]]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dif = np.abs(shots[keys[i]].astype(np.int16)
                         - shots[keys[j]].astype(np.int16)).mean()
            print(f"          {keys[i]} vs {keys[j]}: {dif:.3f}/255", flush=True)
    print("[scale] and matched vs controls (must be LARGE):")
    for c in [r[0] for r in rows[3:]]:
        dif = np.abs(shots[keys[0]].astype(np.int16)
                     - shots[c].astype(np.int16)).mean()
        print(f"          {keys[0]} vs {c}: {dif:.3f}/255", flush=True)

    np.savez_compressed(args.out, **shots,
                        table=np.array(rows, dtype=object), allow_pickle=True)
    print(f"\n[scale] wrote {args.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
