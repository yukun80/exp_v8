import torch
import torch.nn as nn
from ..modules.conv import Conv
from ..modules.block import C2f
from .attention import SEAttention
from .mamba_vss import VSSBlock

__all__ = ["ContextGuideFusionModule", "C2f_LVMB"]


class ContextGuideFusionModule(nn.Module):
    """ContextGuideFusionModule"""

    def __init__(self, inc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = SEAttention(inc[1] * 2)

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(
            x_concat, [x0.size()[1], x1.size()[1]], dim=1
        )
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class C2f_LVMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))
