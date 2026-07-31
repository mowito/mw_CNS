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
    """Self-attention over the fused tokens, WITH 2D axial RoPE.

    It had none. `nn.MultiheadAttention` carries no positional information, and the
    action token then collapses all N tokens to one vector, so the whole controller
    was EXACTLY permutation invariant over the grid -- measured on a trained
    checkpoint: shuffling the token order of all three streams together gave
    cos(base, shuffled) = 1.000000 (Sec. 6.24).

    That mattered most for the fine CNN branch. Position can still reach the head
    through token CONTENT -- RADIO's absolute embeddings ride inside F_c, and P's
    sparsity pattern encodes absolute patch position -- but the fine branch is a plain
    conv stack whose ONLY positional signal is where its features sit on the grid.
    So the branch the paper adds "to capture the pixel-wise error to improve the servo
    precision" had its spatial information discarded by construction, which is the
    leading explanation for a precision floor that no data-side fix moved.

    Reuses refine.MHAttention rather than a second RoPE implementation; spec Sec. 3.6
    annotates exactly these blocks with RoPE.
    """

    def __init__(self, dim, heads=8, use_rope=True):
        super().__init__()
        self.use_rope = use_rope
        self.n1 = nn.LayerNorm(dim)
        if use_rope:
            from .refine import MHAttention
            self.attn = MHAttention(dim, heads, use_rope=True)
        else:
            # Deliberately the ORIGINAL nn.MultiheadAttention, not MHAttention with
            # use_rope=False. The two have different parameter names (in_proj/out_proj
            # vs q/k/v/proj), so routing the no-RoPE path through MHAttention would
            # make every pre-existing checkpoint unloadable. Keeping this branch byte-
            # compatible means ctrl_rope=False genuinely reproduces the old model and
            # the earlier runs stay evaluable.
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, hw):
        y = self.n1(x)
        if self.use_rope:
            x = x + self.attn(y, y, hw, hw)
        else:
            x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class NeuralController(nn.Module):
    def __init__(self, feat_dim: int, grid_dim: int, dim: int = 256,
                 n_self: int = 3, heads: int = 8, regress_norm: bool = True,
                 fine_dim: int = 0, ctrl_rope: bool = True):
        super().__init__()
        self.regress_norm = regress_norm
        self.fine_dim = fine_dim
        # project coarse current features and the probability grid into token space.
        # P is a distribution over grid_dim=K*K cells, so its entries sit at ~1/K^2
        # (~4e-3) while the refined ViT features are LayerNorm'd to std~1. Feeding P
        # in raw makes its contribution ~1e-3 of the feature contribution -- swamped
        # by grid_proj's own bias -- so the controller ignores the correspondence
        # signal entirely and can only regress the mean velocity. Normalize the grid
        # to feature scale first; this keeps the *pattern* (which cells are likely)
        # and only discards the meaningless absolute magnitude.
        self.feat_proj = nn.Linear(feat_dim, dim)
        self.grid_norm = nn.LayerNorm(grid_dim)
        self.grid_proj = nn.Linear(grid_dim, dim)
        # Third stream: fine-grained CNN features (paper Fig. 2 / cns/models/fine_cnn.py).
        # Fig. 2 concatenates the CNN output with P before the self-attention stack;
        # Eq. 10 additionally names F_c, so all three streams are fused here. Set
        # fine_dim=0 to reproduce the coarse-only ablation.
        n_stream = 3 if fine_dim else 2
        if fine_dim:
            self.fine_norm = nn.LayerNorm(fine_dim)
            self.fine_proj = nn.Linear(fine_dim, dim)
        self.fuse = nn.Linear(n_stream * dim, dim)
        self.ctrl_rope = ctrl_rope
        self.self_blocks = nn.ModuleList(
            [_SelfBlock(dim, heads, use_rope=ctrl_rope) for _ in range(n_self)])
        # learned action token cross-attends into the fused tokens (grid-conditioned)
        self.action_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross = _CrossAttn(dim, heads)
        self.cross_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dir_head = nn.Linear(dim, 6)                 # v~_dir
        self.norm_head = nn.Linear(dim, 1) if regress_norm else None   # l~

    def forward(self, Fc: torch.Tensor, P: torch.Tensor, hidden=None, fine=None):
        """Fc: [B,H,W,feat_dim] (refined current features), P: [B,H,W,grid_dim],
        fine: [B,H,W,fine_dim] CNN fine-grained features or None.
        Returns (vec [B,6], log_norm [B,1] or None, hidden(passthrough))."""
        B, H, W, _ = Fc.shape
        tf = self.feat_proj(Fc.reshape(B, H * W, -1))
        tg = self.grid_proj(self.grid_norm(P.reshape(B, H * W, -1)))
        streams = [tf, tg]
        if self.fine_dim:
            if fine is None:
                raise ValueError(
                    "controller was built with fine_dim>0 but no fine features were "
                    "passed -- the CNN branch needs the raw image pair (Fig. 2)")
            streams.append(self.fine_proj(self.fine_norm(fine.reshape(B, H * W, -1))))
        tok = self.fuse(torch.cat(streams, dim=-1))         # [B,N,dim]
        for blk in self.self_blocks:
            tok = blk(tok, (H, W))          # RoPE needs the grid shape
        act = self.action_token.expand(B, 1, -1)
        act = act + self.cross(act, tok)                   # cross-attend into grid-conditioned tokens
        act = self.cross_norm(act).squeeze(1)              # [B,dim]
        feat = self.head(act)
        vec = self.dir_head(feat)                          # [B,6]
        log_norm = self.norm_head(feat) if self.regress_norm else None
        return vec, log_norm, hidden

    # ---- CNS-v1 trainer contract -------------------------------------------
    @staticmethod
    def postprocess(raw_pred, tPo_norm, max_mag=3.0):
        """(vec, log_norm) -> real-world-scaled velocity [B,6] (Eq. 22 + depth
        re-scale of translation). tPo_norm: [B] or [B,1] scene scale s=d*.

        max_mag clamps the UNIT-WORLD magnitude before the scale is reapplied.
        `sigma(x) = exp(x-1) if x <= 1 else x` is LINEAR above 1, so the head's
        magnitude output passes through unbounded -- there is no saturation to lean
        on. One outlier prediction becomes an enormous metric velocity. Measured in
        gate 6:
        504.7 mm -> 7223.9 mm in THREE steps at dt=1/50, i.e. ~112 m/s, which threw
        the camera clear of the workspace and ended an otherwise recoverable episode
        (Sec. 6.21). A single blown step is unrecoverable because there is no
        velocity limit anywhere else in the loop.

        3.0 is well above anything legitimate: the label ||vel_si|| is ~TE/d* for
        the translation part, so a unit-world magnitude of 3 already means "move
        three times the scene distance in one second". Clamping the magnitude
        preserves the DIRECTION, which is the part the policy is good at
        (per-bin cos 0.77-0.92). Set max_mag=0 to disable.
        """
        vec, log_norm, _ = raw_pred
        direction = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
        mag = sigma(log_norm) if log_norm is not None else vec.norm(dim=-1, keepdim=True)
        if max_mag and max_mag > 0:
            mag = mag.clamp(max=max_mag)
        v = direction * mag                                # normalized (unit-world) velocity
        s = tPo_norm.view(-1, 1)
        v = v.clone()
        v[:, :3] = v[:, :3] * s                            # Eq. 19-20: scale translation
        return v

    @staticmethod
    def objectives(raw_pred, vel_si, norm_weight: float = 1.0):
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
        # Eq.23 sums the two terms unweighted, which is fine when they are the
        # same order. With near-goal data they are not: sigma_inv(y)=1+log(y)
        # reaches -3.9 for y~0.007, so l_norm runs ~1.09 against l_dir ~0.69 and
        # supplies ~60% of the gradient. Direction is what drives servo
        # convergence, so allow the magnitude term to be down-weighted.
        loss = l_dir + norm_weight * l_norm
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
