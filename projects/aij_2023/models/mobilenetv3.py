import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from torchvision.utils import _log_api_usage_once
from torchvision.models._utils import _make_divisible


__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]

class PositionalEncoding(nn.Module):
    def __init__(self, in_dims, dropout = 0.0, max_len = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_dims, 2) * (-math.log(10000.0) / in_dims))
        pe = torch.zeros(1, max_len, in_dims)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class TemporalAttnetion(nn.Module):
    def __init__(self, frame_count, dim=64, token_dim=256, num_heads=2):
        super().__init__()
        self.frame_count = frame_count
        self.to_value = nn.Linear(dim, token_dim * num_heads)
        self.to_query = nn.Linear(dim, token_dim * num_heads)
        self.to_key = nn.Linear(dim, token_dim * num_heads)

        self.scale_factor = token_dim ** -0.5
        self.to_out = nn.Linear(token_dim * num_heads, dim)

        self.pos_encoder = PositionalEncoding(dim)

    def forward(self, x):
        bf, c, h, w = x.shape
        b = bf // self.frame_count
        x = x.unsqueeze(1).reshape(b, self.frame_count, c, h, w).contiguous()
        x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.frame_count, c).contiguous()

        x = self.pos_encoder(x)
        residual = x

        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxFxC
        key = torch.nn.functional.normalize(key, dim=-1) #BxFxC
        value = torch.nn.functional.normalize(value, dim=-1) #BxFxC

        mask = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)

        attention_scores = torch.baddbmm(
            mask,
            query,
            key.transpose(-1, -2),
            beta=0.0,
            alpha=1.0,
        )
        del mask
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        x = torch.bmm(attention_probs, value)
        x = self.to_out(x)
        x = x + residual

        x = x.reshape(b, h, w, self.frame_count, c).permute(0, 4, 3, 1, 2).contiguous()
        x = x.reshape(bf, 1, c, h, w).squeeze(1).contiguous()
        return x


class Conv3DBlock(nn.Module):
    def __init__(self, channels, frame_count, pool_kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.frame_count = frame_count
        self.norm = nn.InstanceNorm3d(num_features=channels, eps=1e-6, affine=True)
        self.conv = nn.Conv3d(
            channels,
            channels,
            pool_kernel,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )
    
    def forward(self, x):
        residual = x
        bf, c, h, w = x.shape
        b = bf // self.frame_count
        x = x.unsqueeze(1).reshape(b, self.frame_count, c, h, w).contiguous()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        x = self.norm(x)
        x = self.conv(x)
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(b*self.frame_count, 1, c, h, w).squeeze(1).contiguous()
        x = x + residual
        return x

class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        frame_count: int = 32,
        temporal_indices: list = None,
        conv3d_indices: list = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.InstanceNorm2d, affine=True)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for layer_index, cnf in enumerate(inverted_residual_setting):
            layers.append(block(cnf, norm_layer))
            if layer_index in temporal_indices:
                layers.append(TemporalAttnetion(frame_count=frame_count, dim=cnf.out_channels))
            if layer_index in conv3d_indices:
                layers.append(Conv3DBlock(channels=cnf.out_channels, frame_count=frame_count))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        # lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            # Conv2dNormActivation(
            #     lastconv_input_channels,
            #     lastconv_output_channels,
            #     kernel_size=1,
            #     norm_layer=norm_layer,
            #     activation_layer=nn.Hardswish,
            # )
            Conv3DBlock(channels=lastconv_input_channels, frame_count=frame_count)
        )

        self.features = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1568, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).contiguous()

        x = self.features(x)
        # x = self.avgpool(x)
        
        _, c_, h_, w_ = x.shape
        x = x.unsqueeze(1).reshape(b, c_, f, h_, w_).contiguous()
        # print(x.shape)
        x = x.flatten(2).mean(1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


# def mobilenet_v3(arch, **kwargs) -> MobileNetV3:
#     inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)
#     model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
#     return model