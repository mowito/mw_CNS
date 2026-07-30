"""Offline probe of a trained checkpoint, by ||vel_si|| bin -- answers Sec. 9.1.

Two questions the closed-loop SR alone cannot separate:
  1. Is the predicted DIRECTION accurate per error-magnitude bin? (near-goal
     chance-level cosine was the recorded failure of the 4060 runs)
  2. Is the near-goal direction estimate UNBIASED? Zero-mean noise still
     converges under a proportional law; a systematic bias stalls at the bias.
     Measured as the norm of the mean unit-error vector: 0 = unbiased.
Also reports the magnitude ratio sigma(l)/||v*||, since a systematic overshoot
destabilizes the loop even with perfect directions.

    python scripts/probe_policy.py --ckpt checkpoints/cnsv2.pth \
        --cache checkpoints/cnsv2_feat
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cns.models.cnsv2_net import build_from_checkpoint
from cns.models.controller import sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cnsv2.pth")
    ap.add_argument("--cache", default="checkpoints/cnsv2_feat")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=1500)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    net = build_from_checkpoint(ck, dev).eval()

    meta = torch.load(os.path.join(args.cache, "meta.pt"), map_location="cpu")
    Fc = np.load(os.path.join(args.cache, "fc.npy"), mmap_mode="r")
    Fd = np.load(os.path.join(args.cache, "fd.npy"), mmap_mode="r")
    Ic = np.load(os.path.join(args.cache, "ic.npy"), mmap_mode="r")
    Id = np.load(os.path.join(args.cache, "id.npy"), mmap_mode="r")
    vsi, va, sid = meta["vsi"], meta["va"][:args.max_val], meta["scene_id"]

    pred = []
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            idx = va[i:i + args.batch]
            fc = torch.from_numpy(np.ascontiguousarray(Fc[idx.numpy()])).to(dev).float()
            fd = torch.from_numpy(np.ascontiguousarray(Fd[sid[idx].numpy()])).to(dev).float()
            ic = torch.from_numpy(np.ascontiguousarray(Ic[idx.numpy()])).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
            idd = torch.from_numpy(np.ascontiguousarray(Id[sid[idx].numpy()])).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
            vec, log_norm, _ = net.head_forward(fc, fd, Ic=ic, Id=idd)
            d = vec / vec.norm(dim=-1, keepdim=True)
            m = sigma(log_norm).squeeze(-1)
            pred.append(torch.cat([d, m[:, None]], dim=-1).cpu())
    pred = torch.cat(pred)                     # [N,7]: unit direction + magnitude
    gt = vsi[va]
    gt_n = gt.norm(dim=-1)
    gt_d = gt / gt_n[:, None].clamp_min(1e-9)

    cosv = (pred[:, :6] * gt_d).sum(-1)
    mag_ratio = pred[:, 6] / gt_n.clamp_min(1e-9)

    print(f"checkpoint: {args.ckpt} (iters_done={ck.get('iters_done')})")
    print(f"val pairs probed: {len(va)}")
    print(f"\n{'||vel_si|| bin':>14} {'n':>5} {'dir cos':>8} {'|bias|':>7} "
          f"{'mag p50':>8} {'mag p90':>8}")
    edges = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gt_n >= lo) & (gt_n < hi)
        if m.sum() < 5:
            continue
        # bias: mean of the unit ERROR vectors (pred_dir - gt_dir, normalized).
        # For zero-mean directional noise this norm -> 0 with n; a systematic
        # bias leaves it O(1).
        err = pred[m, :6] - gt_d[m]
        err = err / err.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        bias = float(err.mean(0).norm())
        lbl = f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},inf)"
        print(f"{lbl:>14} {int(m.sum()):>5} {float(cosv[m].mean()):>8.4f} "
              f"{bias:>7.3f} {float(mag_ratio[m].median()):>8.3f} "
              f"{float(mag_ratio[m].quantile(0.9)):>8.3f}")
    print(f"\noverall: dir cos {float(cosv.mean()):.4f}, "
          f"l_dir {float((1-cosv).mean()):.4f}, "
          f"mag ratio p50 {float(mag_ratio.median()):.3f}")
    print("read: dir cos near 1 = good; |bias| near 0 = unbiased (Sec 9.1); "
          "mag p50/p90 near 1 = calibrated, >>1 = overshoot risk in closed loop")


if __name__ == "__main__":
    main()
