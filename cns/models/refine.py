"""CNSv2 cross-image feature refinement transformer (paper Eq. 13).

    {F_bar_c, F_bar_d} = Transformer(F_c, F_d)

Interleaved self-attention (within each image) and cross-attention (between the
two images), following LoFTR / GMFlow practice. Positional encoding is 2D axial
RoPE (paper's explicit choice over learned/sinusoidal, for cross-resolution
generalization). Structure is a lightweight port of the RoMa-style refiner, not
its pretrained weights.

Shapes: features are handled as [B, N, C] with N = H16*W16, and the (H16, W16)
grid shape is passed alongside so RoPE knows the 2D layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Axial2DRoPE(nn.Module):
    """2D axial rotary embedding. Splits per-head channels into an x-group and a
    y-group; each is rotated by its axis coordinate. head_dim must be % 4 == 0."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D axial RoPE"
        self.head_dim = head_dim
        self.base = base
        self._cache = {}

    def _emb(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self._cache:
            q = self.head_dim // 4                       # freqs per axis
            freqs = 1.0 / (self.base ** (torch.arange(q, device=device, dtype=torch.float32) / q))
            h = torch.arange(H, device=device, dtype=torch.float32)
            w = torch.arange(W, device=device, dtype=torch.float32)
            h_ang = torch.outer(h, freqs)[:, None, :].expand(H, W, q)   # [H,W,q]
            w_ang = torch.outer(w, freqs)[None, :, :].expand(H, W, q)   # [H,W,q]
            ang = torch.cat([w_ang, h_ang], dim=-1).reshape(H * W, self.head_dim // 2)
            emb = torch.cat([ang, ang], dim=-1)          # [N, head_dim]
            self._cache[key] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return self._cache[key]

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, heads, N, head_dim]
        cos, sin = self._emb(H, W, x.device, x.dtype)
        return x * cos + rotate_half(x) * sin


class MHAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, use_rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = Axial2DRoPE(self.head_dim) if use_rope else None

    def _split(self, t, B, N):
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,N,d]

    def forward(self, x_q, x_kv, hw_q, hw_kv):
        """x_q: [B,Nq,C] queries; x_kv: [B,Nk,C] keys/values (== x_q for self-attn)."""
        B, Nq, C = x_q.shape
        Nk = x_kv.shape[1]
        q = self._split(self.q(x_q), B, Nq)
        k = self._split(self.k(x_kv), B, Nk)
        v = self._split(self.v(x_kv), B, Nk)
        if self.rope is not None:
            q = self.rope(q, hw_q[0], hw_q[1])
            k = self.rope(k, hw_kv[0], hw_kv[1])
        out = F.scaled_dot_product_attention(q, k, v)      # [B,h,Nq,d]
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class RefineBlock(nn.Module):
    """One self-attn (each image independently) + one cross-attn (c<->d) round,
    pre-norm residual, shared weights across the two images."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.self_attn = MHAttention(dim, num_heads)
        self.n2 = nn.LayerNorm(dim); self.cross_attn = MHAttention(dim, num_heads)
        self.n3 = nn.LayerNorm(dim); self.ffn = FFN(dim)

    def forward(self, fc, fd, hw):
        # self-attention within each image
        fc = fc + self.self_attn(self.n1(fc), self.n1(fc), hw, hw)
        fd = fd + self.self_attn(self.n1(fd), self.n1(fd), hw, hw)
        # cross-attention between images
        nc, nd = self.n2(fc), self.n2(fd)
        fc = fc + self.cross_attn(nc, nd, hw, hw)
        fd = fd + self.cross_attn(nd, nc, hw, hw)
        # feed-forward
        fc = fc + self.ffn(self.n3(fc))
        fd = fd + self.ffn(self.n3(fd))
        return fc, fd


class RefineTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([RefineBlock(dim, num_heads) for _ in range(num_layers)])

    def forward(self, Fc: torch.Tensor, Fd: torch.Tensor):
        """Fc, Fd: [B, H16, W16, C]  ->  refined [B, H16, W16, C] each."""
        B, H, W, C = Fc.shape
        fc = Fc.reshape(B, H * W, C)
        fd = Fd.reshape(B, H * W, C)
        for blk in self.blocks:
            fc, fd = blk(fc, fd, (H, W))
        return fc.reshape(B, H, W, C), fd.reshape(B, H, W, C)
