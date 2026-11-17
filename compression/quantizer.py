"""From-scratch Weight & Activation Quantization + BatchNorm Folding for MobileNet-v2.

Implements:
1. Exact BatchNorm2d-into-Conv2d folding (`fold_batch_norm_into_conv`).
2. Per-Output-Channel Asymmetric K-bit Weight Quantization with uint8 sub-byte bit-packing
   (`QuantizedConv2d`, `QuantizedLinear`) that deletes original FP32 parameters from RAM/VRAM.
3. Configurable K-bit Activation Quantization (`ActivationQuantizer`) with percentile outlier
   clipping and real-time activation memory tracking.

No external compression libraries or torch.ao.quantization APIs are used.
"""

import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.bitpack import pack_bits, unpack_bits, packed_nbytes


@torch.no_grad()
def quantize_per_channel_asymmetric(
    weight: torch.Tensor,
    bits: int,
    clip_percentile: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizes a weight tensor of shape (C_out, ...) per output channel to [0, 2^bits - 1].

    Args:
        weight: Float32 tensor of shape (C_out, C_in/groups, kH, kW) or (C_out, C_in).
        bits: Bit-width in [2, 8].
        clip_percentile: Optional percentile in (99.0, 100.0] to clip extreme channel outliers
            at very low bit-widths (<= 4 bits).

    Returns:
        q_weight: Unsigned integer tensor in [0, 2^bits - 1] of same shape as `weight`.
        scale: FP16 tensor of shape (C_out,) storing per-channel scale factors.
        zero_point: uint8 tensor of shape (C_out,) storing per-channel zero points.
    """
    assert 2 <= bits <= 8, f"Weight bit-width must be in [2, 8], got {bits}"
    c_out = weight.shape[0]
    flat = weight.view(c_out, -1).float()
    qmin = 0.0
    qmax = float((1 << bits) - 1)

    if clip_percentile < 100.0 and flat.shape[1] >= 16:
        alpha = (100.0 - clip_percentile) / 200.0
        w_min = torch.quantile(flat, alpha, dim=1)
        w_max = torch.quantile(flat, 1.0 - alpha, dim=1)
    else:
        w_min = flat.min(dim=1).values
        w_max = flat.max(dim=1).values

    # Ensure zero is representable in the range [w_min, w_max]
    w_min = torch.minimum(w_min, torch.zeros_like(w_min))
    w_max = torch.maximum(w_max, torch.zeros_like(w_max))

    scale = (w_max - w_min) / max(qmax - qmin, 1.0)
    scale = torch.clamp(scale, min=1e-8)

    zero_point = torch.round(-w_min / scale).clamp(qmin, qmax).to(torch.uint8)
    scale_fp16 = scale.to(torch.float16)

    # Use rounded FP16 scale during quantization so dequantization matches exactly
    scale_used = scale_fp16.float().unsqueeze(1)
    zp_used = zero_point.float().unsqueeze(1)

    q_flat = torch.round(flat / scale_used + zp_used).clamp(qmin, qmax).to(torch.uint8)
    return q_flat.view(weight.shape), scale_fp16, zero_point


class ActivationQuantizer(nn.Module):
    """Configurable K-bit dynamic activation quantizer with bit-packing memory accounting.

    Quantizes intermediate feature maps to `bits` unsigned integers in [0, 2^bits - 1].
    Also tracks uncompressed (FP32) vs compressed (bit-packed + FP16 scale + uint8 zero_point)
    activation memory footprint during forward passes.
    """

    def __init__(
        self,
        bits: int = 8,
        clip_percentile: float = 99.95,
        pack_in_memory: bool = False,
        layer_name: str = "",
    ) -> None:
        super().__init__()
        self.bits = bits
        self.clip_percentile = clip_percentile
        self.pack_in_memory = pack_in_memory
        self.layer_name = layer_name

        # Runtime telemetry updated on forward pass
        self.last_numel: int = 0
        self.last_fp32_bytes: int = 0
        self.last_compressed_bytes: int = 0
        self.last_metadata_bytes: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_numel = x.numel()
        self.last_fp32_bytes = x.numel() * 4  # FP32 = 4 bytes/element

        if self.bits >= 32 or self.bits <= 0:
            self.last_compressed_bytes = self.last_fp32_bytes
            self.last_metadata_bytes = 0
            return x

        qmin = 0.0
        qmax = float((1 << self.bits) - 1)

        with torch.no_grad():
            # Per-tensor dynamic range for activations (1 FP16 scale + 1 uint8 zero_point = 3 bytes)
            x_min = x.min()
            x_max = x.max()
            if self.clip_percentile < 100.0 and self.bits <= 4:
                # Slight outlier clipping for ultra-low bit activations (e.g., linear bottlenecks)
                flat_sample = x.reshape(-1)
                if flat_sample.numel() > 4096:
                    stride = flat_sample.numel() // 4096
                    flat_sample = flat_sample[::stride]
                alpha = (100.0 - self.clip_percentile) / 200.0
                x_min = torch.quantile(flat_sample, alpha)
                x_max = torch.quantile(flat_sample, 1.0 - alpha)

            x_min = torch.minimum(x_min, torch.zeros_like(x_min))
            x_max = torch.maximum(x_max, torch.zeros_like(x_max))

            scale = torch.clamp((x_max - x_min) / qmax, min=1e-8).to(torch.float16)
            zp = torch.round(-x_min / scale.float()).clamp(qmin, qmax).to(torch.uint8)

            q_x = torch.round(x / scale.float() + zp.float()).clamp(qmin, qmax).to(torch.uint8)

            if self.pack_in_memory:
                packed, orig_shape = pack_bits(q_x, self.bits)
                q_x_unpacked = unpack_bits(packed, orig_shape, self.bits, out_dtype=x.dtype)
            else:
                q_x_unpacked = q_x.to(x.dtype)

            dequant_x = (q_x_unpacked - zp.to(x.dtype)) * scale.to(x.dtype)

            # Record exact bit-packed activation buffer footprint + metadata overhead (3 bytes)
            self.last_metadata_bytes = 3  # 2B (fp16 scale) + 1B (uint8 zero_point)
            self.last_compressed_bytes = (
                packed_nbytes(x.numel(), self.bits) + self.last_metadata_bytes
            )

        return dequant_x


class QuantizedConv2d(nn.Module):
    """In-memory compressed Conv2d storing bit-packed uint8 weights and FP16 metadata.

    Replaces `nn.Conv2d` (after BatchNorm folding). The original FP32 weight parameter is
    deleted from memory; only `packed_weight` (uint8), `scale` (fp16), `zero_point` (uint8),
    and `bias` (fp16) reside in RAM/VRAM.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        weight_bits: int = 8,
        activation_bits: int = 8,
        clip_percentile: float = 100.0,
        pack_activations: bool = False,
        layer_name: str = "",
    ) -> None:
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight_bits = weight_bits
        self.layer_name = layer_name
        self.weight_shape = tuple(conv.weight.shape)
        self.weight_numel = conv.weight.numel()

        q_w, scale, zp = quantize_per_channel_asymmetric(
            conv.weight.detach(), bits=weight_bits, clip_percentile=clip_percentile
        )
        packed_w, _ = pack_bits(q_w, bits=weight_bits)

        # Register compressed buffers in RAM/VRAM (no nn.Parameter float32 weights!)
        self.register_buffer("packed_weight", packed_w)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zp)

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().to(torch.float16))
        else:
            self.register_buffer("bias", None)

        self.act_quantizer = ActivationQuantizer(
            bits=activation_bits,
            pack_in_memory=pack_activations,
            layer_name=f"{layer_name}.act",
        )

    def unpack_dequantize_weight(self) -> torch.Tensor:
        """Unpacks uint8 buffer and dequantizes per-channel weights on-the-fly for matmul."""
        q_w = unpack_bits(
            self.packed_weight,
            self.weight_shape,
            self.weight_bits,
            out_dtype=torch.float32,
        )
        scale = self.scale.float().view(-1, 1, 1, 1)
        zp = self.zero_point.float().view(-1, 1, 1, 1)
        return (q_w - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quantizer(x)
        w_dq = self.unpack_dequantize_weight()
        bias = self.bias.float() if self.bias is not None else None
        return F.conv2d(
            x_q,
            w_dq,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def memory_breakdown(self) -> Dict[str, int]:
        """Returns exact in-memory byte counts for packed weights and metadata overheads."""
        packed_bytes = self.packed_weight.numel() * self.packed_weight.element_size()
        scale_bytes = self.scale.numel() * self.scale.element_size()
        zp_bytes = self.zero_point.numel() * self.zero_point.element_size()
        bias_bytes = (
            self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        )
        # Shape & config scalar metadata (out_c, in_c, kH, kW, bits = 5 x int16 = 10 bytes)
        struct_bytes = 10
        return {
            "fp32_weight_bytes": self.weight_numel * 4,
            "packed_weight_bytes": packed_bytes,
            "scale_bytes": scale_bytes,
            "zp_bytes": zp_bytes,
            "bias_bytes": bias_bytes,
            "struct_bytes": struct_bytes,
            "total_metadata_bytes": scale_bytes + zp_bytes + bias_bytes + struct_bytes,
            "total_compressed_bytes": (
                packed_bytes + scale_bytes + zp_bytes + bias_bytes + struct_bytes
            ),
        }


class QuantizedLinear(nn.Module):
    """In-memory compressed Linear layer storing bit-packed uint8 weights and FP16 metadata."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_bits: int = 8,
        activation_bits: int = 8,
        clip_percentile: float = 100.0,
        pack_activations: bool = False,
        layer_name: str = "",
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_bits = weight_bits
        self.layer_name = layer_name
        self.weight_shape = tuple(linear.weight.shape)
        self.weight_numel = linear.weight.numel()

        q_w, scale, zp = quantize_per_channel_asymmetric(
            linear.weight.detach(), bits=weight_bits, clip_percentile=clip_percentile
        )
        packed_w, _ = pack_bits(q_w, bits=weight_bits)

        self.register_buffer("packed_weight", packed_w)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zp)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16))
        else:
            self.register_buffer("bias", None)

        self.act_quantizer = ActivationQuantizer(
            bits=activation_bits,
            pack_in_memory=pack_activations,
            layer_name=f"{layer_name}.act",
        )

    def unpack_dequantize_weight(self) -> torch.Tensor:
        q_w = unpack_bits(
            self.packed_weight,
            self.weight_shape,
            self.weight_bits,
            out_dtype=torch.float32,
        )
        scale = self.scale.float().view(-1, 1)
        zp = self.zero_point.float().view(-1, 1)
        return (q_w - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quantizer(x)
        w_dq = self.unpack_dequantize_weight()
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x_q, w_dq, bias=bias)

    def memory_breakdown(self) -> Dict[str, int]:
        packed_bytes = self.packed_weight.numel() * self.packed_weight.element_size()
        scale_bytes = self.scale.numel() * self.scale.element_size()
        zp_bytes = self.zero_point.numel() * self.zero_point.element_size()
        bias_bytes = (
            self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        )
        struct_bytes = 6
        return {
            "fp32_weight_bytes": self.weight_numel * 4,
            "packed_weight_bytes": packed_bytes,
            "scale_bytes": scale_bytes,
            "zp_bytes": zp_bytes,
            "bias_bytes": bias_bytes,
            "struct_bytes": struct_bytes,
            "total_metadata_bytes": scale_bytes + zp_bytes + bias_bytes + struct_bytes,
            "total_compressed_bytes": (
                packed_bytes + scale_bytes + zp_bytes + bias_bytes + struct_bytes
            ),
        }


@torch.no_grad()
def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuses a Conv2d and BatchNorm2d pair into a single Conv2d with bias."""
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(conv.weight.device)

    w = conv.weight.clone().float()
    mean = bn.running_mean.float()
    var_sqrt = torch.sqrt(bn.running_var.float() + bn.eps)
    gamma = bn.weight.float()
    beta = bn.bias.float()

    if conv.bias is not None:
        b = conv.bias.clone().float()
    else:
        b = torch.zeros_like(mean)

    factor = gamma / var_sqrt
    fused_conv.weight.copy_(w * factor.view(-1, 1, 1, 1))
    fused_conv.bias.copy_((b - mean) * factor + beta)
    return fused_conv


@torch.no_grad()
def fold_batch_norm_into_conv(model: nn.Module) -> nn.Module:
    """Recursively folds all adjacent (Conv2d, BatchNorm2d) pairs in MobileNet-v2."""
    model = copy.deepcopy(model)
    model.eval()

    for _, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for idx in range(len(module) - 1):
                if isinstance(module[idx], nn.Conv2d) and isinstance(
                    module[idx + 1], nn.BatchNorm2d
                ):
                    module[idx] = _fuse_conv_and_bn(module[idx], module[idx + 1])
                    module[idx + 1] = nn.Identity()
    return model


def _replace_layers_recursive(
    module: nn.Module,
    weight_bits: int,
    activation_bits: int,
    clip_percentile: float,
    pack_activations: bool,
    prefix: str = "",
) -> None:
    """Recursively replaces all Conv2d and Linear modules with Quantized equivalents."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d):
            setattr(
                module,
                name,
                QuantizedConv2d(
                    child,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    clip_percentile=clip_percentile,
                    pack_activations=pack_activations,
                    layer_name=full_name,
                ),
            )
        elif isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                QuantizedLinear(
                    child,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    clip_percentile=clip_percentile,
                    pack_activations=pack_activations,
                    layer_name=full_name,
                ),
            )
        else:
            _replace_layers_recursive(
                child,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                clip_percentile=clip_percentile,
                pack_activations=pack_activations,
                prefix=full_name,
            )


@torch.no_grad()
def compress_mobilenetv2(
    model: nn.Module,
    weight_quant_bits: int = 8,
    activation_quant_bits: int = 8,
    fold_bn: bool = True,
    clip_percentile: float = 100.0,
    pack_activations: bool = False,
) -> nn.Module:
    """Applies BatchNorm folding and K-bit Weight + Activation Compression to MobileNet-v2.

    Args:
        model: Pretrained FP32 MobileNetV2 instance.
        weight_quant_bits: Target weight bit-width (2..8, or 32 for uncompressed).
        activation_quant_bits: Target activation bit-width (2..8, or 32 for uncompressed).
        fold_bn: Whether to fold BatchNorm2d layers into preceding Conv2d layers first.
        clip_percentile: Percentile clipping threshold for weight quantization.
        pack_activations: Whether to explicitly bit-pack/unpack activations in forward pass.

    Returns:
        Compressed MobileNetV2 model with bit-packed uint8 buffers in memory.
    """
    if weight_quant_bits >= 32 and activation_quant_bits >= 32:
        return copy.deepcopy(model).eval()

    compressed = fold_batch_norm_into_conv(model) if fold_bn else copy.deepcopy(model).eval()

    if weight_quant_bits < 32 or activation_quant_bits < 32:
        w_bits = min(weight_quant_bits, 8)
        a_bits = activation_quant_bits
        _replace_layers_recursive(
            compressed,
            weight_bits=w_bits,
            activation_bits=a_bits,
            clip_percentile=clip_percentile,
            pack_activations=pack_activations,
        )

    return compressed.eval()
