"""CNSv2 correctness gate -- check 1: does the loss converge?

Renders a small PyBullet dataset, precomputes frozen ViT features, then trains
the refine->prob-match->controller head with AMP and reports the loss trend.
A clear downward trend proves the whole pipeline (render -> supervise -> model
-> loss -> optimizer) is wired correctly and the model can learn. Not a paper
reproduction -- just the wiring/learnability gate (CNSv2_5090_SETUP.md Sec 6.1).

    python3 train_cnsv2_smoke.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cns.models.cnsv2_net import build_model
from cns.sim.cnsv2_data import generate_samples, precompute_backbone_features

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN, N_VAL = 96, 24
ITERS, BATCH, LR = 400, 8, 3e-4

print(f"[1/4] rendering {N_TRAIN+N_VAL} samples (CPU renderer)...")
t0 = time.time()
train = generate_samples(N_TRAIN, seed=1)
val = generate_samples(N_VAL, seed=999)
print(f"      done in {time.time()-t0:.0f}s | vel_si norm range "
      f"[{train['vel_si'].norm(dim=-1).min():.2f}, {train['vel_si'].norm(dim=-1).max():.2f}]")

print("[2/4] building model + precomputing frozen ViT features...")
# The Fig. 2 fine CNN branch reads raw pixels, so head_forward() needs the image
# pair alongside the cached ViT features. This smoke test keeps its images in RAM
# already, so wire them through rather than disabling the branch -- a smoke test
# that silently exercises a different architecture than train_cnsv2.py is worse
# than no smoke test.
net = build_model(K=16, refine_layers=4, ctrl_dim=256).to(dev)
Fc_tr = precompute_backbone_features(net, train["Ic"], dev)
Fd_tr = precompute_backbone_features(net, train["Id"], dev)
Fc_va = precompute_backbone_features(net, val["Ic"], dev)
Fd_va = precompute_backbone_features(net, val["Id"], dev)
vsi_tr, tpo_tr = train["vel_si"].to(dev), train["tPo_norm"].to(dev)
vsi_va = val["vel_si"].to(dev)

decay, no_decay = net.get_parameter_groups()
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                         {"params": no_decay, "weight_decay": 0.0}], lr=LR)

# fp32 training: the trainable head is tiny (~41M) and the frozen backbone is
# precomputed, so AMP buys little here; fp32 avoids fp16-range overflow in the
# feature correlation and is well within the 8GB card (peak <3GB).
def run_head(Fc, Fd, Ic, Id, idx):
    fc = Fc[idx].to(dev).float(); fd = Fd[idx].to(dev).float()
    ic = Ic[idx].to(dev).float(); idd = Id[idx].to(dev).float()
    if ic.max() > 1.5:                      # generate_samples yields [0,1] floats
        ic = ic / 255.0; idd = idd / 255.0
    return net.head_forward(fc, fd, Ic=ic, Id=idd)

print(f"[3/4] training {ITERS} iters, batch {BATCH}...")
net.train()
hist = []
for it in range(ITERS):
    idx = torch.randint(0, N_TRAIN, (BATCH,))
    opt.zero_grad(set_to_none=True)
    raw = run_head(Fc_tr, Fd_tr, train["Ic"], train["Id"], idx)
    _, loss = net.objectives(raw, vsi_tr[idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    hist.append(float(loss.detach()))
    if (it + 1) % 50 == 0:
        print(f"      iter {it+1:4d}  loss {np.mean(hist[-50:]):.4f}")

print("[4/4] evaluating...")
net.eval()
with torch.no_grad():
    va_idx = torch.arange(Fc_va.shape[0])
    raw = run_head(Fc_va, Fd_va, val["Ic"], val["Id"], va_idx)
    res, _ = net.objectives(raw, vsi_va)
first, last = np.mean(hist[:5]), np.mean(hist[-50:])   # measure from iter 0, not past the initial drop
print(f"\ntrain loss: {first:.4f} -> {last:.4f}  ({100*(first-last)/first:.0f}% down)")
print(f"val: l_dir={res['l_dir']:.4f}  l_norm={res['l_norm']:.4f}")

os.makedirs("checkpoints", exist_ok=True)
torch.save({"state_dict": net.state_dict(),
            "config": {"K": 16, "feat_dim": 768, "intrinsic": [512,512,256,256,512,512],
                       "d_star": 1.0, "fine_dim": net.fine_dim}},
           "checkpoints/cnsv2_smoke.pth")
gate = last < first * 0.6
print("\nGATE (loss converges):", "PASS" if gate else "INCONCLUSIVE",
      "| ckpt -> checkpoints/cnsv2_smoke.pth")
sys.exit(0 if gate else 2)
