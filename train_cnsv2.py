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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="checkpoints/cnsv2.pth")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[load] {args.data}", flush=True)
    d = load_bproc_samples(args.data)
    n = d["Ic"].shape[0]
    nv = max(8, int(n * args.val_frac))
    perm = torch.randperm(n)
    tr, va = perm[nv:], perm[:nv]
    print(f"[load] {n} samples -> train {len(tr)} / val {len(va)}", flush=True)

    net = build_model(K=CONFIG["K"], refine_layers=4, ctrl_dim=256).to(dev)
    print("[feat] precomputing frozen ViT features...", flush=True)
    Fc = precompute_backbone_features(net, d["Ic"], dev)   # CPU fp16
    Fd = precompute_backbone_features(net, d["Id"], dev)
    vsi = d["vel_si"]

    decay, no_decay = net.get_parameter_groups()
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    def batch_forward(idx):
        fc = Fc[idx].to(dev).float(); fd = Fd[idx].to(dev).float()
        return net.head_forward(fc, fd)

    @torch.no_grad()
    def validate():
        net.eval()
        raw = batch_forward(va)
        res, _ = net.objectives(raw, vsi[va].to(dev))
        net.train()
        return res

    def save(tag):
        torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                    "iters_done": it + 1}, args.out.replace(".pth", f"_{tag}.pth"))

    print(f"[train] {args.iters} iters, batch {args.batch}, fp32", flush=True)
    net.train(); t0 = time.time(); hist = []
    for it in range(args.iters):
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
            print(f"  iter {it+1:6d}/{args.iters}  train {np.mean(hist[-args.log_every:]):.4f}  "
                  f"val l_dir {v['l_dir']:.4f} l_norm {v['l_norm']:.4f}  "
                  f"{rate:.1f} it/s", flush=True)
        if (it + 1) % args.save_every == 0:
            save(f"iter{it+1}")

    torch.save({"state_dict": net.state_dict(), "config": CONFIG,
                "iters_done": args.iters}, args.out)
    v = validate()
    print(f"[done] {(time.time()-t0)/3600:.2f} h | final val l_dir {v['l_dir']:.4f} "
          f"l_norm {v['l_norm']:.4f} | ckpt -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
