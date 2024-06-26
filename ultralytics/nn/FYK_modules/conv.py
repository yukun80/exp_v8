import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import CBAM


__all__ = ["ConvX", "DepthwiseSeparableConv", "DepthwiseSeparableCBAMConv"]


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel,
                stride=stride,
                dilation=dilation,
                padding=dilation,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积,用于对Concat之后的特征进行降维。
    首先通过depthwise卷积处理空间信息，
    使用了与输入通道数相同数量的分组，每组中的卷积核负责处理一个通道的信息（depthwise卷积）.
    然后通过pointwise卷积进行降维,
    使用1x1的卷积核将这些特征合并，并调整通道数到所需的输出通道数（pointwise卷积）。
    """

    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseSeparableCBAMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(DepthwiseSeparableCBAMConv, self).__init__()
        self.cbam = CBAM(in_channels, kernel_size)
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels)

    def forward(self, x):
        x = self.cbam(x)
        x = self.depthwise_separable_conv(x)
        return x


if __name__ == "__main__":
    """python -m ultralytics.nn.FYK_modules.conv"""
    # model = ConvX(3, 64)
    # x = torch.randn(2, 3, 224, 224)
    # out = model(x)
    # print(out.shape)

    # model = DepthwiseSeparableConv(64, 128)
    # x = torch.randn(2, 64, 56, 56)
    # out = model(x)
    # print(out.shape)

    model = DepthwiseSeparableCBAMConv(64, 32)
    input_tensor = torch.randn(2, 64, 128, 128)
    out = model(input_tensor)
    print(input_tensor.shape, out.shape)
