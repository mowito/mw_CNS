"""CNSv2 full policy: image pair -> probability grid -> normalized velocity.

    F_c, F_d   = ViT(I_c), ViT(I_d)                 backbone.py   (Eq. 12)
    F_bar_*    = Transformer(F_c, F_d)              refine.py     (Eq. 13)
    S, P       = ProbMatch(F_bar_c, F_bar_d)        prob_match.py (Eq. 14-17)
    vec, l~    = NeuralController(F_bar_c, P)        controller.py (Eq. 10, 22)

Exposes the CNS-v1 trainer contract (forward -> (vec, log_norm, hidden),
postprocess / objectives / get_parameter_groups) so it plugs into
cns/utils/trainer.py with the graph model swapped out.
"""

import torch
import torch.nn as nn

from .backbone import build_backbone
from .refine import RefineTransformer
from ..midend.prob_match import ProbMatch
from .controller import NeuralController
from .fine_cnn import FineCNN

VITB_DIM = 768   # AM-RADIO ViT-B spatial feature dim


class CNSv2Net(nn.Module):
    def __init__(self, K: int = 16, feat_dim: int = VITB_DIM, refine_layers: int = 4,
                 refine_heads: int = 8, ctrl_dim: int = 256, ctrl_self: int = 3,
                 backbone_version: str = "radio_v2.5-b", freeze_backbone: bool = True,
                 regress_norm: bool = True, load_backbone: bool = True,
                 fine_dim: int = 128, ctrl_rope: bool = True):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.fine_dim = fine_dim
        self.backbone = build_backbone(backbone_version, freeze=freeze_backbone) if load_backbone else None
        self.refine = RefineTransformer(dim=feat_dim, num_layers=refine_layers, num_heads=refine_heads)
        self.prob = ProbMatch(K=K)
        # Fig. 2's fine-grained CNN branch. Trainable, and unlike the coarse ViT
        # features it cannot be cached, because it needs the raw image pair.
        self.fine = FineCNN(out_dim=fine_dim) if fine_dim else None
        self.controller = NeuralController(
            feat_dim=feat_dim, grid_dim=K * K, dim=ctrl_dim, n_self=ctrl_self,
            regress_norm=regress_norm, fine_dim=fine_dim, ctrl_rope=ctrl_rope)

    def features(self, Ic, Id):
        """Run frozen backbone -> refined features (allows caching desired-image feats)."""
        Fc, Fd = self.backbone(Ic), self.backbone(Id)
        assert Fc.shape[-1] == self.feat_dim, \
            f"backbone dim {Fc.shape[-1]} != configured feat_dim {self.feat_dim}"
        return self.refine(Fc, Fd)

    def head_forward(self, Fc, Fd, Ic=None, Id=None, hidden=None):
        """Run the TRAINABLE head (refine -> prob-match -> [+fine CNN] -> controller)
        on precomputed RAW (frozen) backbone features Fc,Fd: [B,H16,W16,C].
        Lets training cache the frozen ViT features once and skip re-running it.

        Ic,Id are still required when fine_dim>0: the CNN branch is trainable and
        reads pixels, so it cannot be folded into the frozen feature cache."""
        rfc, rfd = self.refine(Fc, Fd)
        out = self.prob(rfc, rfd)
        fine = None
        if self.fine is not None:
            if Ic is None or Id is None:
                raise ValueError(
                    "head_forward needs Ic,Id when fine_dim>0 (Fig. 2 CNN branch); "
                    "build the model with fine_dim=0 for the coarse-only ablation")
            fine = self.fine(Ic, Id)
        return self.controller(rfc, out["P"], hidden, fine=fine)

    @torch.no_grad()
    def backbone_features(self, Ic, Id):
        """Frozen ViT features (no refine). [B,3,H,W] -> two [B,H16,W16,C]."""
        return self.backbone(Ic), self.backbone(Id)

    def forward(self, Ic, Id=None, hidden=None, feats=None):
        """Ic,Id: [B,3,H,W] in [0,1]. Or pass precomputed feats=(rfc,rfd) -- but the
        fine CNN branch always needs the images, so feats alone is only valid when
        fine_dim=0.
        Returns (vec [B,6], log_norm [B,1] or None, hidden(passthrough))."""
        rfc, rfd = feats if feats is not None else self.features(Ic, Id)
        out = self.prob(rfc, rfd)
        fine = self.fine(Ic, Id) if self.fine is not None else None
        return self.controller(rfc, out["P"], hidden, fine=fine)

    # ---- trainer contract (delegates to controller) ------------------------
    @staticmethod
    def postprocess(raw_pred, tPo_norm):
        return NeuralController.postprocess(raw_pred, tPo_norm)

    @staticmethod
    def objectives(raw_pred, vel_si, norm_weight: float = 1.0):
        return NeuralController.objectives(raw_pred, vel_si, norm_weight)

    def get_parameter_groups(self):
        """Only trainable params (frozen backbone excluded via requires_grad)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or "action_token" in name or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay


def build_model(**kw) -> CNSv2Net:
    return CNSv2Net(**kw)


def build_from_checkpoint(ck, device="cuda", **overrides) -> CNSv2Net:
    """Build a net whose architecture matches a checkpoint, then load it.

    Every inference script used to hardcode `build_model(K=16, refine_layers=4,
    ctrl_dim=256)` and rely on the constructor defaults happening to match what
    was trained. That is fragile in exactly one direction that matters: `fine_dim`
    is now a real architectural switch (`--fine-dim 0` reproduces the coarse-only
    ablation), so a checkpoint trained one way and evaluated the other either
    crashes inside load_state_dict or, worse, silently loads a differently shaped
    model.

    The shapes are read from the STATE DICT, not the config blob, because the
    state dict cannot disagree with itself and older checkpoints predate the
    config key entirely.

        ck  = torch.load(path, map_location=dev)
        net = build_from_checkpoint(ck, dev)
    """
    sd = ck["state_dict"] if "state_dict" in ck else ck
    cfg = dict(ck.get("config") or {}) if isinstance(ck, dict) else {}

    kw = {"K": cfg.get("K", 16), "refine_layers": 4, "ctrl_dim": 256}
    if "feat_dim" in cfg:
        kw["feat_dim"] = cfg["feat_dim"]

    # fine_dim: present iff the Fig. 2 CNN branch was trained in.
    if "controller.fine_proj.weight" in sd:
        kw["fine_dim"] = int(sd["controller.fine_proj.weight"].shape[1])
    else:
        kw["fine_dim"] = 0

    # K from the grid projection, which is K*K wide -- authoritative over cfg.
    if "controller.grid_proj.weight" in sd:
        grid_dim = int(sd["controller.grid_proj.weight"].shape[1])
        kw["K"] = int(round(grid_dim ** 0.5))
    if "controller.action_token" in sd:
        kw["ctrl_dim"] = int(sd["controller.action_token"].shape[-1])
    kw["refine_layers"] = 1 + max(
        (int(k.split(".")[2]) for k in sd if k.startswith("refine.blocks.")),
        default=kw["refine_layers"] - 1)

    # ctrl_rope: read from the state dict, never assumed. The controller's self-attn
    # switched from nn.MultiheadAttention (no positional encoding, so exactly
    # permutation invariant -- Sec. 6.24) to refine.MHAttention with 2D axial RoPE.
    # The two have different parameter names, so guessing wrong is a load failure at
    # best and a silently different model at worst.
    kw["ctrl_rope"] = any(k.startswith("controller.self_blocks.0.attn.q.")
                          for k in sd)

    kw.update(overrides)
    net = CNSv2Net(**kw).to(device)
    net.load_state_dict(sd)
    print(f"[model] built from checkpoint: K={kw['K']} ctrl_dim={kw['ctrl_dim']} "
          f"refine_layers={kw['refine_layers']} fine_dim={kw['fine_dim']} "
          f"ctrl_rope={kw['ctrl_rope']}"
          + ("  (NO fine CNN branch -- coarse-only ablation)" if not kw["fine_dim"] else ""),
          flush=True)
    return net
