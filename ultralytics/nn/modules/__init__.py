# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.
当你使用 from module import * 语句时，__all__ 变量定义了哪些名称会被导入到当前的命名空间中。
Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    Silence,
)

from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    Segment,
    Detect,
    WorldDetect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "TransformerEncoderLayer",
    "RepC3",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
)
