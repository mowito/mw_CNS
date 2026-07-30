"""Fine-grained CNN branch -- paper Fig. 2 (bottom-left) and Sec. I.

Fig. 2's bottom pipeline is

    [I_c, I_d] -> Concat -> CNN -> Concat(with P) -> SelfAttn -> CrossAttn -> MLP

and its caption states the purpose outright: "Fine-grained features from CNN are
also fused to capture the pixel-wise error to improve the servo precision."
Sec. I says why it is a CNN and not more transformer: "instead we use only several
low-cost convolution layers to fuse fine-grained features, which significantly
improves the model efficiency."

WHY THIS MATTERS HERE: the coarse branch is ViT/16, so one token spans 16 px. At
the paper's canonical d*=1 m that is 31.2 mm of object displacement per patch,
while Table I row 1 reports TE = 0.948 mm -- about 1/33 of a patch. The coarse
correspondence grid cannot represent that error; this branch is where sub-patch
information enters the controller. It was missing from our implementation
entirely, which is the most likely reason the policy sat at chance near the goal
(measured translation cosine 0.51 in the near-goal bin).

Four stride-2 convolutions take 512x512 -> 32x32, matching the ViT/16 token grid
so the two streams and P can be concatenated per token. GroupNorm rather than
BatchNorm because the paper trains at batch 16 and DAgger rollouts run batch 1.
"""

import torch
import torch.nn as nn


class FineCNN(nn.Module):
    """concat(I_c, I_d) -> per-token fine features at the ViT/16 grid.

    in:  [B, 3, H, W] x2, pixels in [0,1]
    out: [B, H/16, W/16, out_dim]
    """

    def __init__(self, out_dim: int = 128, width: int = 32, groups: int = 8):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 3

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(min(groups, cout), cout),
                nn.GELU(),
            )

        # 6 channels in: the two images are concatenated on the channel axis, so
        # the very first conv can already form a per-pixel difference feature --
        # that is the "pixel-wise error" the figure caption refers to.
        self.net = nn.Sequential(
            block(6, c1, 2),        # /2
            block(c1, c2, 2),       # /4
            block(c2, c3, 2),       # /8
            block(c3, out_dim, 2),  # /16  -> matches ViT/16 tokens
        )
        self.out_dim = out_dim

    def forward(self, Ic: torch.Tensor, Id: torch.Tensor) -> torch.Tensor:
        if Ic.dtype == torch.uint8:
            Ic = Ic.float().div(255.0)
        if Id.dtype == torch.uint8:
            Id = Id.float().div(255.0)
        x = torch.cat([Ic, Id], dim=1)          # [B,6,H,W]
        f = self.net(x)                          # [B,out_dim,H16,W16]
        return f.permute(0, 2, 3, 1).contiguous()   # [B,H16,W16,out_dim]
