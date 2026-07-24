"""End-to-end test of the assembled CNSv2Net with the real frozen RADIO backbone.
Reports peak VRAM at the canonical 512x512 to size the training batch on 8GB.
    python3 tests/smoke_full_net.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from cns.models.cnsv2_net import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, HW, K = 2, 512, 16
net = build_model(K=K, refine_layers=4, ctrl_dim=256).to(dev)
net.train()

n_train = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
n_froz = sum(p.numel() for p in net.parameters() if not p.requires_grad) / 1e6
print(f"trainable={n_train:.1f}M  frozen={n_froz:.1f}M")

Ic = torch.rand(B, 3, HW, HW, device=dev)
Id = torch.rand(B, 3, HW, HW, device=dev)
tPo = torch.rand(B, device=dev) + 0.5
vel_si = torch.randn(B, 6, device=dev)

if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()

# mixed precision, as planned for the 8GB card
with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev == "cuda")):
    raw = net(Ic, Id)
    vel = net.postprocess(raw, tPo)
    res, loss = net.objectives(raw, vel_si)
loss.backward()

print("forward vec/log_norm shapes:", tuple(raw[0].shape), tuple(raw[1].shape))
print("postprocessed vel:", tuple(vel.shape), "loss:", res)
bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.backbone.parameters())
tr_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.refine.parameters())
print("grad flows to refine/controller:", tr_grad, "| backbone frozen (no grad):", not bb_grad)
if dev == "cuda":
    print(f"PEAK VRAM (B={B}, 512x512, K={K}, fp16 fwd+bwd): "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print("RESULT: FULL NET OK")
