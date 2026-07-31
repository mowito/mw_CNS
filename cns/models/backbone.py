"""CNSv2 frozen foundation feature extractor (paper Eq. 12).

    F_c = ViT(I_c),  F_d = ViT(I_d)

AM-RADIOv2.5 with a ViT-B structure, patch size 16, frozen (no fine-tuning).
Input  I in R^{H1 x W1 x 3} (here [B,3,H1,W1], pixels in [0,1]).
Output F in R^{H16 x W16 x C}, H16=H1/16, W16=W1/16, C=768 for ViT-B.

Weights are loaded from TorchHub (NVlabs/RADIO). Note: original RADIO weights
are NSCLv1 / non-commercial -- fine for the prototype phase; swap in a C-RADIO
variant or DINOv2 (Apache-2.0) with the same frozen ViT-B interface if this ever
needs to be commercial-safe (see CNSv2_5090_SETUP.md Sec 4.1).
"""

import torch
import torch.nn as nn


class RadioBackbone(nn.Module):
    PATCH = 16

    def __init__(self, version: str = "radio_v2.5-b", freeze: bool = True):
        super().__init__()
        self.version = version
        # skip_validation avoids a GitHub API rate-limit check on torch.hub.
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version=version,
            progress=True, skip_validation=True, trust_repo=True,
        )
        # Cropped Position Embedding is what makes RADIO's ABSOLUTE positional
        # embeddings resolution-robust, and enable_cpe() is a monkey-patch applied on
        # top of a stock timm ViT. Load by any route that skips it and you silently
        # get the backbone WITHOUT CPE -- no error, no warning, just worse features at
        # any resolution other than the one it was pretrained at. Verified True for
        # the torch.hub route above; asserted so a loader change cannot regress it
        # unnoticed. Nested two levels: RADIOModel.model is the timm ViT.
        pg = getattr(getattr(self.model, "model", None), "patch_generator", None)
        if pg is None:
            print("[backbone] WARNING: no patch_generator found; cannot verify "
                  "cpe_mode. If RADIO's internals moved, re-check that Cropped "
                  "Position Embedding is active.", flush=True)
        elif not getattr(pg, "cpe_mode", False):
            raise SystemExit(
                "[backbone] REFUSING to run: patch_generator.cpe_mode is False, so "
                "this RADIO was loaded WITHOUT Cropped Position Embedding. Absolute "
                "position embeddings are then not resolution-robust and every "
                "feature downstream is degraded silently.")

        self.freeze = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        # Discover the spatial feature channel count once, lazily (see forward).
        self._embed_dim = None

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            raise RuntimeError("embed_dim is known only after the first forward()")
        return self._embed_dim

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode regardless of parent .train().
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _forward_model(self, img: torch.Tensor):
        out = self.model(img)
        # RADIO returns (summary, spatial_features) or an object exposing them.
        if isinstance(out, (tuple, list)):
            summary, feat = out[0], out[1]
        else:
            summary, feat = out.summary, out.features
        return summary, feat

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B,3,H,W] in [0,1]  ->  F: [B, H16, W16, C]."""
        B, _, H, W = img.shape
        assert H % self.PATCH == 0 and W % self.PATCH == 0, \
            f"input H,W must be multiples of {self.PATCH}, got {(H, W)}"
        H16, W16 = H // self.PATCH, W // self.PATCH

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            _, feat = self._forward_model(img)   # feat: [B, H16*W16, C]

        C = feat.shape[-1]
        self._embed_dim = C
        assert feat.shape[1] == H16 * W16, \
            f"RADIO returned {feat.shape[1]} tokens, expected {H16 * W16}"
        return feat.reshape(B, H16, W16, C).contiguous()


def build_backbone(version: str = "radio_v2.5-b", freeze: bool = True) -> RadioBackbone:
    return RadioBackbone(version=version, freeze=freeze)
