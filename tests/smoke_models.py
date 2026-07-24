"""Shape/gradient/math smoke test for the CNSv2 model modules.
Runs on random feature maps -- no RADIO download or GPU required.
    python3 tests/smoke_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cns.models.refine import RefineTransformer
from cns.midend.prob_match import ProbMatch, bspline_quadratic
from cns.models.controller import NeuralController, sigma, sigma_inv
from cns.models import denorm
import numpy as np

torch.manual_seed(0)
ok = True
def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("== B-spline kernel (Eq. 17) ==")
a = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
k = bspline_quadratic(a)
check("kappa(0)=0.75", abs(k[0] - 0.75) < 1e-6)
check("kappa(0.5)=0.5", abs(k[1] - 0.5) < 1e-6)
check("kappa(1.0)=0.125", abs(k[2] - 0.125) < 1e-6)
check("kappa(1.5)=0", abs(k[3]) < 1e-6)
check("kappa(2.0)=0", abs(k[4]) < 1e-6)
# partition of unity: sum of kappa over integer shifts == 1
xs = torch.linspace(-0.49, 0.49, 7)
part = sum(bspline_quadratic(xs - n) for n in (-1, 0, 1))
check("partition of unity ~1", torch.allclose(part, torch.ones_like(part), atol=1e-5))

print("== sigma round-trip (Eq. 22) ==")
y = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
check("sigma(sigma_inv(y))==y", torch.allclose(sigma(sigma_inv(y)), y, atol=1e-5))

print("== forward chain: refine -> probmatch -> controller ==")
B, H16, W16, C, K = 2, 8, 8, 64, 4
N, D = H16 * W16, K * K
Fc = torch.randn(B, H16, W16, C, requires_grad=True)
Fd = torch.randn(B, H16, W16, C, requires_grad=True)
refine = RefineTransformer(dim=C, num_layers=2, num_heads=4)
pm = ProbMatch(K=K)
ctrl = NeuralController(feat_dim=C, grid_dim=D, dim=64, n_self=2, heads=4)

rfc, rfd = refine(Fc, Fd)
check("refine keeps shape", rfc.shape == (B, H16, W16, C))
out = pm(rfc, rfd, return_explicit=True)
S, P = out["S"], out["P"]
check("S shape [B,N,N]", S.shape == (B, N, N))
check("S rows sum to 1 (softmax)", torch.allclose(S.sum(-1), torch.ones(B, N), atol=1e-5))
check("P shape [B,H,W,D]", P.shape == (B, H16, W16, D))
check("P finite", torch.isfinite(P).all())
check("explicit corr shape [B,N,2]", out["corr"].shape == (B, N, 2))

vec, log_norm, _ = ctrl(rfc, P)
check("vec shape [B,6]", vec.shape == (B, 6))
check("log_norm shape [B,1]", log_norm.shape == (B, 1))

print("== postprocess + losses (Eq. 20, 23) ==")
tPo = torch.tensor([0.8, 1.2])
vel = NeuralController.postprocess((vec, log_norm, None), tPo)
check("postprocess vel [B,6]", vel.shape == (B, 6))
vel_si = torch.randn(B, 6)
res, loss = NeuralController.objectives((vec, log_norm, None), vel_si)
check("loss is scalar & finite", loss.dim() == 0 and torch.isfinite(loss))
loss.backward()
g_ctrl = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ctrl.parameters())
g_ref = any(p.grad is not None and p.grad.abs().sum() > 0 for p in refine.parameters())
check("grad flows to controller", g_ctrl)
check("grad flows to refine transformer (through P and S)", g_ref)

print("== denormalization (Eq. 18-21) ==")
vn = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
vr = denorm.denormalize_scale(vn, d_star=0.5)
check("scale only affects translation", np.allclose(vr[3:], vn[3:]) and np.allclose(vr[:3], vn[:3] * 0.5))
R, t = denorm.pbvs_inverse(vn)
check("pbvs_inverse R is SO(3)", np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6)
Kc = np.array([[512, 0, 256], [0, 512, 256], [0, 0, 1]], float)
Kr = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], float)
vi = denorm.denormalize_intrinsic(vn, Kr, Kc, d_star=0.5)
check("intrinsic-denorm returns finite (6,)", vi.shape == (6,) and np.all(np.isfinite(vi)))
# canonical==canonical should reduce to plain scale (up to pose recovery numerics)
vi0 = denorm.denormalize_intrinsic(vn, Kc, Kc, d_star=0.5)
check("intrinsic-denorm ~ scale when K==canonical", np.allclose(vi0, vr, atol=1e-6))

print("\nPARAM COUNTS:",
      f"refine={sum(p.numel() for p in refine.parameters())/1e3:.0f}k,",
      f"controller={sum(p.numel() for p in ctrl.parameters())/1e3:.0f}k")
print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
