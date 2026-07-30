"""Measure Eq. 25 gravity-centre quality against ground truth, by error bin.

The closed-loop failure chain hypothesis (2026-07-30): far from the goal, feature
loss degrades the dual-softmax matching, so the NETWORK gravity centres are
garbage; the hybrid law's rotation then steers off the scene, features vanish
entirely, and both laws (and the raw policy) operate on out-of-distribution
images with nothing to bound the excursion. This probe tests the first link
directly: net-estimated cXg/dXg vs the ground-truth projected object centroid,
per ||vel_si|| bin, on training-domain scenes.

Also audits the Sec. G SWITCH: with GT centres, should_use_hybrid says hybrid
far / PBVS near. If the NET centres flip that decision on a large fraction of
far-field pairs (S ~ uniform => cXg ~ dXg ~ image centre), the deployment
switches to the raw policy exactly where the paper wants the safe law.

    python scripts/probe_gravity.py --data data/isaac_train --n-scenes 120
"""
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cns.models.cnsv2_net import build_from_checkpoint
from cns.models.hybrid_control import should_use_hybrid, PATCH_PX
from cns.sim.supervisor import supervisor_vel, Policy
from cns.utils.perception import CameraIntrinsic

FX = FY = 512.0
CX = CY = 256.0
IMG = 512


def gt_gravity_patch(wcT, wP):
    Rw, t = wcT[:3, :3], wcT[:3, 3]
    pc = (wP - t) @ Rw
    z = pc[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return None
    u = FX * pc[ok, 0] / z[ok] + CX
    v = FY * pc[ok, 1] / z[ok] + CY
    m = (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    if m.sum() == 0:
        return None
    return np.array([u[m].mean(), v[m].mean()]) / PATCH_PX - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--data", default="data/isaac_train")
    ap.add_argument("--n-scenes", type=int, default=120)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    net = build_from_checkpoint(ck, dev).eval()
    intr = CameraIntrinsic(512, 512, 512, 512, 256, 256)
    dummy, dz = np.zeros((1, 2)), np.ones(1)

    files = sorted(glob.glob(os.path.join(args.data, "scene_*.npz")))[:args.n_scenes]
    rows = []      # (||vel_si||, cErr, dErr, gt_gap, net_gap, gt_hyb, net_hyb)
    with torch.no_grad():
        for f in files:
            z = np.load(f)
            imgs, poses, wP = z["images"], z["poses"], z["wP"]
            Id_t = torch.from_numpy(imgs[0]).permute(2, 0, 1)[None].to(dev).float() / 255.0
            gt_d = gt_gravity_patch(poses[0], wP)
            if gt_d is None:
                continue
            for i in range(1, imgs.shape[0]):
                gt_c = gt_gravity_patch(poses[i], wP)
                if gt_c is None:
                    continue
                _, (tpo, vsi) = supervisor_vel(Policy.PBVS_Straight, dummy, dz,
                                               dummy, dz, intr, poses[i], poses[0], wP)
                Ic_t = torch.from_numpy(imgs[i]).permute(2, 0, 1)[None].to(dev).float() / 255.0
                rfc, rfd = net.features(Ic_t, Id_t)
                cXg, dXg, _ = net.prob.gravity_centers(rfc, rfd)
                cXg = cXg[0].cpu().numpy(); dXg = dXg[0].cpu().numpy()
                rows.append((float(np.linalg.norm(vsi)),
                             float(np.linalg.norm(cXg - gt_c)),
                             float(np.linalg.norm(dXg - gt_d)),
                             float(np.linalg.norm(np.asarray(gt_c) - np.asarray(gt_d))),
                             float(np.linalg.norm(cXg - dXg)),
                             should_use_hybrid(gt_c, gt_d),
                             should_use_hybrid(cXg, dXg)))

    r = np.array(rows, float)
    print(f"pairs probed: {len(r)}   (errors in PATCH units; 1 patch = 16 px)")
    print(f"\n{'||vel_si|| bin':>14} {'n':>5} {'cXg err':>8} {'dXg err':>8} "
          f"{'gt gap':>7} {'net gap':>8} {'switch agree':>13}")
    edges = [0, 0.25, 0.5, 1.0, 2.0, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.sum() < 5:
            continue
        agree = float((r[m, 5] == r[m, 6]).mean())
        lbl = f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},inf)"
        print(f"{lbl:>14} {int(m.sum()):>5} {np.median(r[m,1]):>8.2f} "
              f"{np.median(r[m,2]):>8.2f} {np.median(r[m,3]):>7.2f} "
              f"{np.median(r[m,4]):>8.2f} {100*agree:>12.0f}%")
    print("\nread: cXg/dXg err ~ how wrong Eq. 25 estimates are; if net gap << gt gap "
          "in far bins, uniform matching is collapsing both centres to the image "
          "centre and the Sec. G switch fires at exactly the wrong range.")


if __name__ == "__main__":
    main()
