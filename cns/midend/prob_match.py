"""CNSv2 probabilistic matching representation -- the core novel piece
(paper Eq. 14-17).

Pipeline:
  Eq.14  S_{c->d} = softmax(F_bar_c . F_bar_d^T / sqrt(C))        [B, N, N]
  Eq.15  x^i_{c->d} = sum_j S^{i,j} x^j_d   (explicit corr, DEBUG only)
  Eq.16  P_{h,w,j} = [sum_k S^{i,k} kappa((f^{i,k}-x_a)/g)]        [B, H16, W16, D]
                     / [sum_k kappa((f^{i,k}-x_a)/g)]
  Eq.17  kappa = quadratic B-spline (see kernel below)

P is the translation-equivariant, resolution-agnostic grid that actually
conditions the controller (NOT the explicit correspondence). Particles are the
entries of S at flow position f^{i,k} = x_d^k - x_c^i with quantity S^{i,k};
they are splatted onto a fixed K x K anchor grid (D = K*K), so the channel
dimension D is independent of image resolution.

Because patch coordinates are fixed, the S -> P map is a fixed linear scatter:
we precompute per-(i,k) anchor indices + kernel weights once and cache them.
(A Toeplitz/conv form is possible since f depends only on the k-i offset;
left as a future optimization -- correctness first.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bspline_quadratic(a: torch.Tensor) -> torch.Tensor:
    """kappa(a), Eq. 17 (1D). Vectorized over any shape."""
    aa = a.abs()
    k = torch.zeros_like(aa)
    m1 = aa < 0.5
    m2 = (aa >= 0.5) & (aa < 1.5)
    k = torch.where(m1, 0.75 - aa * aa, k)
    k = torch.where(m2, 0.5 * (1.5 - aa) ** 2, k)
    return k


def patch_grid_coords(H16: int, W16: int, device) -> torch.Tensor:
    """[N,2] coordinates (x=w, y=h) of each patch, N=H16*W16, index i=h*W16+w."""
    ys, xs = torch.meshgrid(
        torch.arange(H16, device=device, dtype=torch.float32),
        torch.arange(W16, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [N,2]


class ParticleToGrid(nn.Module):
    """Splat score matrix S onto a K x K anchor grid (Eq. 16-17)."""

    def __init__(self, K: int = 16, eps: float = 1e-6):
        super().__init__()
        self.K = K
        self.eps = eps
        self._plan = {}   # (H16,W16,device) -> (idx[N,N], w[N,N]) list over combos

    def _build_plan(self, H16, W16, device):
        K = self.K
        N = H16 * W16
        D = K * K
        grid = patch_grid_coords(H16, W16, device)          # [N,2]
        # flow f[i,k] = x_d[k] - x_c[i]  -> [N, N, 2]
        f = grid[None, :, :] - grid[:, None, :]
        origin = torch.tensor([-W16, -H16], device=device, dtype=torch.float32)
        g = torch.tensor([2.0 * W16 / K, 2.0 * H16 / K], device=device, dtype=torch.float32)
        gc = (f - origin) / g                                # [N,N,2] fractional anchor coords
        base = torch.floor(gc[..., 0]).long(), torch.floor(gc[..., 1]).long()

        idx_list, w_list = [], []
        for oy in (-1, 0, 1, 2):
            for ox in (-1, 0, 1, 2):
                ax = base[0] + ox                            # [N,N] anchor x index
                ay = base[1] + oy                            # [N,N] anchor y index
                arg_x = gc[..., 0] - (ax.float() + 0.5)
                arg_y = gc[..., 1] - (ay.float() + 0.5)
                w = bspline_quadratic(arg_x) * bspline_quadratic(arg_y)   # [N,N]
                valid = (ax >= 0) & (ax < K) & (ay >= 0) & (ay < K) & (w > 0)
                flat = torch.where(valid, ay * K + ax, torch.full_like(ax, D))  # D = dump col
                w = torch.where(valid, w, torch.zeros_like(w))
                idx_list.append(flat)     # [N,N] in [0, D]
                w_list.append(w)          # [N,N]
        # Denominator is constant (independent of B and S) -> precompute once.
        den = torch.zeros(N, D + 1, device=device)
        for idx, w in zip(idx_list, w_list):
            den.scatter_add_(1, idx, w)
        den = den[:, :D].unsqueeze(0)                 # [1,N,D] broadcast over batch
        self._plan[(H16, W16, device)] = (idx_list, w_list, N, D, den)

    def forward(self, S: torch.Tensor, H16: int, W16: int) -> torch.Tensor:
        """S: [B, N, N] (rows=current patches, cols=desired). -> P: [B,H16,W16,D].

        Forced to fp32: the eps in the num/den division underflows to 0 in fp16,
        which turns empty anchors (den=0) into 0/0 = NaN under autocast."""
        device = S.device
        key = (H16, W16, device)
        if key not in self._plan:
            self._build_plan(H16, W16, device)
        idx_list, w_list, N, D, den = self._plan[key]
        B = S.shape[0]
        with torch.autocast(device_type=device.type, enabled=False):
            S = S.float()
            num = S.new_zeros(B, N, D + 1)      # +1 dump column for invalid anchors
            for idx, w in zip(idx_list, w_list):
                idx_b = idx.unsqueeze(0).expand(B, N, N)               # [B,N,N]
                num.scatter_add_(2, idx_b, S * w.unsqueeze(0))
            P = num[..., :D] / (den + self.eps)    # den precomputed & constant
        return P.reshape(B, H16, W16, D)


class ProbMatch(nn.Module):
    """Full probabilistic matching: refined features -> (S, P, [explicit corr])."""

    def __init__(self, K: int = 16):
        super().__init__()
        self.p2g = ParticleToGrid(K=K)
        self.K = K

    def score_matrix(self, Fc: torch.Tensor, Fd: torch.Tensor) -> torch.Tensor:
        """Eq. 14. Fc,Fd: [B,H16,W16,C] -> S: [B,N,N] (softmax over desired).

        Correlation is computed in fp32: the dot product over C=768 dims easily
        exceeds the fp16 range (65504) -> inf -> softmax=NaN. Keeping it fp32
        makes both AMP training and fp16 real-time inference numerically safe."""
        B, H, W, C = Fc.shape
        with torch.autocast(device_type=Fc.device.type, enabled=False):
            fc = Fc.reshape(B, H * W, C).float()
            fd = Fd.reshape(B, H * W, C).float()
            logits = torch.matmul(fc, fd.transpose(1, 2)) / (C ** 0.5)
            return torch.softmax(logits, dim=-1)

    def explicit_corr(self, S: torch.Tensor, H16: int, W16: int) -> torch.Tensor:
        """Eq. 15 (DEBUG/vis only, not fed to controller). -> [B,N,2]."""
        grid = patch_grid_coords(H16, W16, S.device)      # [N,2]
        return torch.matmul(S, grid)                       # [B,N,2]

    def forward(self, Fc: torch.Tensor, Fd: torch.Tensor, return_explicit: bool = False):
        B, H, W, C = Fc.shape
        S = self.score_matrix(Fc, Fd)          # [B,N,N]
        P = self.p2g(S, H, W)                   # [B,H,W,D]
        out = {"S": S, "P": P}
        if return_explicit:
            out["corr"] = self.explicit_corr(S, H, W)
        return out
