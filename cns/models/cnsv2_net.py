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

VITB_DIM = 768   # AM-RADIO ViT-B spatial feature dim


class CNSv2Net(nn.Module):
    def __init__(self, K: int = 16, feat_dim: int = VITB_DIM, refine_layers: int = 4,
                 refine_heads: int = 8, ctrl_dim: int = 256, ctrl_self: int = 3,
                 backbone_version: str = "radio_v2.5-b", freeze_backbone: bool = True,
                 regress_norm: bool = True, load_backbone: bool = True):
        super().__init__()
        self.K = K
        self.feat_dim = feat_dim
        self.backbone = build_backbone(backbone_version, freeze=freeze_backbone) if load_backbone else None
        self.refine = RefineTransformer(dim=feat_dim, num_layers=refine_layers, num_heads=refine_heads)
        self.prob = ProbMatch(K=K)
        self.controller = NeuralController(
            feat_dim=feat_dim, grid_dim=K * K, dim=ctrl_dim, n_self=ctrl_self, regress_norm=regress_norm)

    def features(self, Ic, Id):
        """Run frozen backbone -> refined features (allows caching desired-image feats)."""
        Fc, Fd = self.backbone(Ic), self.backbone(Id)
        assert Fc.shape[-1] == self.feat_dim, \
            f"backbone dim {Fc.shape[-1]} != configured feat_dim {self.feat_dim}"
        return self.refine(Fc, Fd)

    def head_forward(self, Fc, Fd, hidden=None):
        """Run the TRAINABLE head (refine -> prob-match -> controller) on
        precomputed RAW (frozen) backbone features Fc,Fd: [B,H16,W16,C].
        Lets training cache the frozen ViT features once and skip re-running it."""
        rfc, rfd = self.refine(Fc, Fd)
        out = self.prob(rfc, rfd)
        return self.controller(rfc, out["P"], hidden)

    @torch.no_grad()
    def backbone_features(self, Ic, Id):
        """Frozen ViT features (no refine). [B,3,H,W] -> two [B,H16,W16,C]."""
        return self.backbone(Ic), self.backbone(Id)

    def forward(self, Ic, Id=None, hidden=None, feats=None):
        """Ic,Id: [B,3,H,W] in [0,1]. Or pass precomputed feats=(rfc,rfd).
        Returns (vec [B,6], log_norm [B,1] or None, hidden(passthrough))."""
        rfc, rfd = feats if feats is not None else self.features(Ic, Id)
        out = self.prob(rfc, rfd)
        return self.controller(rfc, out["P"], hidden)

    # ---- trainer contract (delegates to controller) ------------------------
    @staticmethod
    def postprocess(raw_pred, tPo_norm):
        return NeuralController.postprocess(raw_pred, tPo_norm)

    @staticmethod
    def objectives(raw_pred, vel_si):
        return NeuralController.objectives(raw_pred, vel_si)

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
