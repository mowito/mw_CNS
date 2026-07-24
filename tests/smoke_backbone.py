"""Live test of the RADIO backbone (downloads ~400MB weights on first run).
    python3 tests/smoke_backbone.py
Validates the TorchHub API + output feature-map shape (the flagged risk).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from cns.models.backbone import build_backbone

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
bb = build_backbone("radio_v2.5-b", freeze=True).to(dev)
img = torch.rand(1, 3, 512, 512, device=dev)
with torch.no_grad():
    F = bb(img)
print("feature map:", tuple(F.shape), "embed_dim:", bb.embed_dim)
assert F.shape[0] == 1 and F.shape[1] == 32 and F.shape[2] == 32, F.shape
print("frozen params:", sum(p.numel() for p in bb.parameters()) / 1e6, "M (requires_grad=%s)" %
      any(p.requires_grad for p in bb.parameters()))
print("RESULT: BACKBONE OK")
