import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv
from ..modules.block import C2f
from .conv import ConvX
from .attention import SEAttention
from .mamba_vss import VSSBlock

__all__ = ["ContextGuideFusionModule", "C2f_LVMB", "MFACB"]


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
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class C2f_LVMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(VSSBlock(self.c) for _ in range(n))


# 一个快速聚拢感受野的方法，改编自STDC
class MFACB(nn.Module):
    """为了不同尺度上学习不同的感知能力
    使用了空洞卷积的Multi-scale Fusion Atrous Convolutional Block (MFACB)模块
    该模块通过使用不同空洞率的卷积层来扩展感受野，从而提高模型的感知能力"""

    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2, 2, 2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes * 3, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # out1 = self.conv_list[0](x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1
