"""CNSv2 neural controller (paper Eq. 10, 22).

    c v~_c = NeuralController(F_c, S)                                   (Eq. 10)
    v~ = sigma(l~) . (v~_dir / ||v~_dir||)                             (Eq. 22)
    sigma(x) = e^{x-1}  if x <= 1 ;   x  if x > 1

Per Fig. 2's bottom pipeline: fine-grained (coarse ViT) features are fused,
passed through self-attention, then a single learned action token cross-attends
into the probability grid P; an MLP head regresses a log-norm scalar l~ and a
6-D direction v~_dir. Output is a 6-D normalized velocity screw [v; w] in the
unit world (denormalized later by cns/models/denorm.py).

The module exposes the CNS-v1 trainer contract so it drops into cns/utils/
trainer.py:  forward(data, hidden)->(vec, log_norm, hidden), and classmethods
postprocess / objectives / get_parameter_groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma(x: torch.Tensor) -> torch.Tensor:
    """Eq. 22 magnitude map (continuous & C1 at x=1)."""
    return torch.where(x <= 1.0, torch.exp(x - 1.0), x)


def sigma_inv(y: torch.Tensor) -> torch.Tensor:
    """Inverse of sigma; used to build the GT log-norm target for L_norm."""
    return torch.where(y <= 1.0, 1.0 + torch.log(y.clamp_min(1e-8)), y)


class _CrossAttn(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.h = heads; self.d = dim // heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, q_tok, kv_tok):
        B, Nq, C = q_tok.shape; Nk = kv_tok.shape[1]
        q = self.q(q_tok).view(B, Nq, self.h, self.d).transpose(1, 2)
        k = self.k(kv_tok).view(B, Nk, self.h, self.d).transpose(1, 2)
        v = self.v(kv_tok).view(B, Nk, self.h, self.d).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, C)
        return self.o(o)


class _SelfBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        y = self.n1(x); x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class NeuralController(nn.Module):
    def __init__(self, feat_dim: int, grid_dim: int, dim: int = 256,
                 n_self: int = 3, heads: int = 8, regress_norm: bool = True):
        super().__init__()
        self.regress_norm = regress_norm
        # project coarse current features and the probability grid into token space
        self.feat_proj = nn.Linear(feat_dim, dim)
        self.grid_proj = nn.Linear(grid_dim, dim)
        self.fuse = nn.Linear(2 * dim, dim)
        self.self_blocks = nn.ModuleList([_SelfBlock(dim, heads) for _ in range(n_self)])
        # learned action token cross-attends into the fused tokens (grid-conditioned)
        self.action_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross = _CrossAttn(dim, heads)
        self.cross_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dir_head = nn.Linear(dim, 6)                 # v~_dir
        self.norm_head = nn.Linear(dim, 1) if regress_norm else None   # l~

    def forward(self, Fc: torch.Tensor, P: torch.Tensor, hidden=None):
        """Fc: [B,H,W,feat_dim] (refined current features), P: [B,H,W,grid_dim].
        Returns (vec [B,6], log_norm [B,1] or None, hidden(passthrough))."""
        B, H, W, _ = Fc.shape
        tf = self.feat_proj(Fc.reshape(B, H * W, -1))
        tg = self.grid_proj(P.reshape(B, H * W, -1))
        tok = self.fuse(torch.cat([tf, tg], dim=-1))       # [B,N,dim]
        for blk in self.self_blocks:
            tok = blk(tok)
        act = self.action_token.expand(B, 1, -1)
        act = act + self.cross(act, tok)                   # cross-attend into grid-conditioned tokens
        act = self.cross_norm(act).squeeze(1)              # [B,dim]
        feat = self.head(act)
        vec = self.dir_head(feat)                          # [B,6]
        log_norm = self.norm_head(feat) if self.regress_norm else None
        return vec, log_norm, hidden

    # ---- CNS-v1 trainer contract -------------------------------------------
    @staticmethod
    def postprocess(raw_pred, tPo_norm):
        """(vec, log_norm) -> real-world-scaled velocity [B,6] (Eq. 22 + depth
        re-scale of translation). tPo_norm: [B] or [B,1] scene scale s=d*."""
        vec, log_norm, _ = raw_pred
        direction = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        mag = sigma(log_norm) if log_norm is not None else vec.norm(dim=-1, keepdim=True)
        v = direction * mag                                # normalized (unit-world) velocity
        s = tPo_norm.view(-1, 1)
        v = v.clone()
        v[:, :3] = v[:, :3] * s                            # Eq. 19-20: scale translation
        return v

    @staticmethod
    def objectives(raw_pred, vel_si):
        """CNSv2 losses (Eq. 23). vel_si: [B,6] GT normalized velocity.
        L_norm = |sigma^-1(||v*||) - l~| (L1);  L_dir = 1 - cos(v*, v~_dir)."""
        vec, log_norm, _ = raw_pred
        vec = vec.float(); vel_si = vel_si.float()
        if log_norm is not None:
            log_norm = log_norm.float()
        gt_norm = vel_si.norm(dim=-1)                                  # [B]
        cos = F.cosine_similarity(vec, vel_si, dim=-1)                 # [B]
        l_dir = (1.0 - cos).mean()
        if log_norm is not None:
            l_norm = (sigma_inv(gt_norm) - log_norm.squeeze(-1)).abs().mean()
        else:
            l_norm = (vec.norm(dim=-1) - gt_norm).abs().mean()
        loss = l_dir + l_norm
        result = {"loss": float(loss.detach()), "l_dir": float(l_dir.detach()),
                  "l_norm": float(l_norm.detach())}
        return result, loss

    def get_parameter_groups(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith("action_token") or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay
