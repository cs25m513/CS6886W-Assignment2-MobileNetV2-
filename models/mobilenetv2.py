"""MobileNet-v2 architecture adapted for CIFAR-10 (32x32 input resolution).

Adapted from PyTorch Vision reference:
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

Key adaptations for CIFAR-10 (32x32 images vs 224x224 ImageNet):
1. Stem Conv2d stride changed from 2 -> 1 (retains 32x32 spatial resolution).
2. Stage 2 (c=24) stride changed from 2 -> 1 (retains 32x32 spatial resolution).
3. Stages 3, 4, 6 use stride=2 (32x32 -> 16x16 -> 8x8 -> 4x4 before global avg pool).
4. Classifier output dimension set to 10 (CIFAR-10 classes).
"""

from typing import List, Optional
import math
import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Ensures that all layer channel counts are divisible by 8 for hardware alignment."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """Standard Conv2d + BatchNorm2d + ReLU6 block used in MobileNet-v2."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[type] = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes, eps=1e-5, momentum=0.1),
            nn.ReLU6(inplace=False),
        )


class InvertedResidual(nn.Module):
    """Inverted Residual with Linear Bottleneck block (Sandler et al., CVPR 2018)."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[type] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], f"Invalid stride {stride}; expected 1 or 2."

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # 1x1 pointwise expansion
            layers.append(
                ConvBNReLU(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    norm_layer=norm_layer,
                )
            )
        layers.extend([
            # 3x3 depthwise separable convolution
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
            ),
            # 1x1 pointwise linear projection (no ReLU6 activation!)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup, eps=1e-5, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNet-v2 network tailored for 32x32 CIFAR-10 images."""

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
        norm_layer: Optional[type] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # [t (expansion), c (channels), n (repeats), s (stride)]
            # Stride in stage 2 (c=24) is 1 instead of 2 to preserve 32x32 feature maps
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # CIFAR-10 adaptation: stride 1 ( stays 32x32 )
                [6, 32, 3, 2],  # Downsamples 32x32 -> 16x16
                [6, 64, 4, 2],  # Downsamples 16x16 -> 8x8
                [6, 96, 3, 1],  # Stays 8x8
                [6, 160, 3, 2], # Downsamples 8x8 -> 4x4
                [6, 320, 1, 1], # Stays 4x4
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        # Stem: stride=1 for CIFAR-10 32x32 input
        features: List[nn.Module] = [
            ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)
        ]

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel

        # Final 1x1 conv expansion to 1280 channels
        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
            )
        )
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2_cifar10(
    num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2
) -> MobileNetV2:
    """Factory function for CIFAR-10 MobileNet-v2."""
    return MobileNetV2(
        num_classes=num_classes, width_mult=width_mult, dropout=dropout
    )
