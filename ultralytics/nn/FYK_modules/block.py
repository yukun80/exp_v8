import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from ..modules.conv import Conv
from ..modules.block import C2f
from .conv import ConvX


__all__ = ["MFACB", "Muti_AFF", "FocalModulation", "EVCBlock", "GLSA"]


##############################MFACB Begin##############################
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
            nn.SiLU(inplace=True),
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes * 3, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.SiLU(inplace=True),
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


##############################MFACB END##############################


##############################MSAF Begin##############################
class Muti_AFF(nn.Module):
    """
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    """

    def __init__(self, chMFACB, chHead, r=4):
        super(Muti_AFF, self).__init__()
        inter_channels = int(chHead // r)

        self.compression = nn.Sequential(
            nn.Conv2d(chMFACB, chHead, kernel_size=1, bias=False),
            nn.BatchNorm2d(chHead, momentum=0.1),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(chHead, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, chHead, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(chHead),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(chHead, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, chHead, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(chHead),
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(chHead, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, chHead, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(chHead),
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(chHead, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, chHead, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(chHead),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chHead, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, chHead, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(chHead),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x[0].shape[2], x[0].shape[3]  # 获取输入 x 的高度和宽度

        residual = self.compression(x[1])
        xa = x[0] + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        if xa.size()[0] == 1:
            print("cao")
            pass
        else:
            xa = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode="nearest")
        c2 = F.interpolate(c2, size=[h, w], mode="nearest")
        c3 = F.interpolate(c3, size=[h, w], mode="nearest")

        xlg = xl + xa + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x[0] * wei + 2 * residual * (1 - wei)
        return xo


##############################MSAF END##############################


class FocalModulation(nn.Module):
    def __init__(
        self,
        dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        bias=True,
        proj_drop=0.0,
        use_postln_in_modulation=False,
        normalize_modulator=False,
    ):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False
                    ),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]

        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l : l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level :]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


# LVC
class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 64
        std = 1.0 / ((num_codes * in_channels) ** 0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True
        )
        # [num_codes]
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        # ---处理codebook (num_code, c1)
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))

        # 把scale从1, num_code变成   batch, c2, N, num_codes
        reshaped_scale = scale.view((1, 1, num_codes))  # N, num_codes

        # ---计算rik = z1 - d  # b, N, num_codes
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()

        # ---处理codebook
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)

        # ---处理特征向量x b, c1, N
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        # 变换rei  b, N, num_codes,-
        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,

        # ---开始计算eik,必须在Rei计算完之后
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


#  1*1 3*3 1*1
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        res_conv=False,
        act_layer=nn.SiLU,
        groups=1,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
        drop_block=None,
        drop_path=None,
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)  # if x_t_r is None else self.conv2(x + x_t_r)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""  # CBL

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class LVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.SiLU(inplace=True),
            Mean(dim=1),
        )
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


# LightMLPBlock
class LightMLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=1,
        stride=1,
        act="silu",
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=nn.GELU,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        drop_path=0.0,
        norm_layer=GroupNorm,
    ):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.out_channels = out_channels

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# EVCBlock
class EVCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = out_channels * expansion
        # Stem stage: get the feature maps by conv block (copied form resnet.py) 进入conformer框架之前的处理
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=1, padding=3, bias=False
        )  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 1 / 4 [56, 56]

        # LVC
        self.lvc = LVCBlock(in_channels=in_channels, out_channels=out_channels, num_codes=64)  # c1值暂时未定
        # LightMLPBlock
        self.l_MLP = LightMLPBlock(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            act="silu",
            act_layer=nn.GELU,
            mlp_ratio=4.0,
            drop=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
            drop_path=0.0,
            norm_layer=GroupNorm,
        )
        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # LVCBlock
        x_lvc = self.lvc(x1)
        # LightMLPBlock
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x


######################################## GLSA begin ########################################
class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=("channel_mul",)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ["avg", "att"]
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ["channel_add", "channel_mul"]
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, "at least one fusion should be used"
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.SiLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.SiLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        try:
            from mmcv.cnn import kaiming_init

            if self.pooling_type == "att":
                kaiming_init(self.conv_mask, mode="fan_in")
                self.conv_mask.inited = True
            # if self.channel_add_conv is not None:
            #     last_zero_init(self.channel_add_conv)
            # if self.channel_mul_conv is not None:
            #     last_zero_init(self.channel_mul_conv)
        except ImportError as e:
            pass

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == "att":
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class GLSASpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(GLSASpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GLSAChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GLSAChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.SiLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class GLSAConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = Conv(in_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv2 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.SiLU(inplace=True))
        self.conv3 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv4 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.SiLU(inplace=True))
        self.conv5 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv6 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.SiLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1, bias=False), nn.SiLU(inplace=True))
        self.ca = GLSAChannelAttention(64)
        self.sa = GLSASpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class GLSA(nn.Module):

    def __init__(self, input_dim=512, embed_dim=32):
        super().__init__()

        self.conv1_1 = Conv(embed_dim * 2, embed_dim, 1)
        self.conv1_1_1 = Conv(input_dim // 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = GLSAConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x_0, x_1 = x.chunk(2, dim=1)

        # local block
        local = self.local(self.local_11conv(x_0))

        # Globel block
        Globel = self.GlobelBlock(self.global_11conv(x_1))

        # concat Globel + local
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)

        return x


######################################## GLSA end ########################################


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    # block = EVCBlock(64, 64).cuda()
    # input = torch.rand(3, 64, 64, 64).cuda()
    # output = block(input)
    # print(input.size(), output.size())

    block1 = Muti_AFF(256, 256)
    input1 = torch.rand(3, 256, 85, 85)
    input2 = torch.rand(3, 256, 85, 85)
    output = block1([input1, input2])
    print(input1.size(), input2.size(), output.size())
