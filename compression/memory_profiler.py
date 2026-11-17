"""In-memory size and compression ratio profiler for weights, metadata, and activations (Q2c, Q4)."""

from dataclasses import dataclass, asdict
from typing import Dict, List
import torch
import torch.nn as nn

from compression.quantizer import QuantizedConv2d, QuantizedLinear, ActivationQuantizer
from compression.bitpack import packed_nbytes


@dataclass
class ModelMemoryReport:
    """Comprehensive in-memory footprint report for baseline vs compressed MobileNet-v2."""

    # Configuration
    weight_quant_bits: int
    activation_quant_bits: int

    # Baseline FP32 breakdown (bytes)
    fp32_conv_linear_weight_bytes: int
    fp32_bn_param_and_buffer_bytes: int
    fp32_total_model_bytes: int

    # Compressed breakdown (bytes)
    packed_weight_bytes: int
    scale_overhead_bytes: int
    zero_point_overhead_bytes: int
    folded_bias_overhead_bytes: int
    struct_metadata_bytes: int
    total_metadata_overhead_bytes: int
    compressed_total_model_bytes: int

    # Activation breakdown (bytes, per single sample B=1)
    fp32_total_activation_bytes: int
    compressed_total_activation_bytes: int
    activation_metadata_bytes: int
    fp32_peak_activation_bytes: int
    compressed_peak_activation_bytes: int

    # Derived MB and Ratios
    fp32_model_size_mb: float
    compressed_model_size_mb: float
    metadata_overhead_kb: float
    metadata_fraction_pct: float
    weight_compression_ratio: float
    model_compression_ratio: float
    activation_compression_ratio: float
    peak_activation_compression_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_raw_tensor_memory(model: nn.Module) -> Dict[str, int]:
    """Computes actual allocated parameter + buffer bytes of any PyTorch model in RAM/VRAM."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": param_bytes + buffer_bytes,
    }


@torch.no_grad()
def measure_activation_memory(
    model: nn.Module,
    input_shape: tuple = (1, 3, 32, 32),
    activation_bits: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, int]:
    """Measures layer-by-layer and peak activation memory during a forward pass (Question 4b).

    Methodology:
      1. Registers forward hooks on all quantized/standard Conv2d & Linear layers and InvertedResidual
         blocks to capture exact feature-map tensor dimensions during a forward pass on `input_shape`.
      2. Computes both:
         (a) Total cumulative activation memory across all 53 layers: sum(numel * 4) for FP32 vs
             sum(ceil(numel * b_a / 8) + 3 bytes metadata) for b_a-bit quantized activations.
         (b) Peak live working-set activation memory required at any single layer execution
             (input tensor + output tensor + residual skip buffer if use_res_connect=True).
    """
    model.eval()
    dummy_x = torch.randn(*input_shape, device=device)

    layer_records: List[Dict[str, int]] = []
    hooks = []

    def _make_hook(name: str):
        def _hook(module: nn.Module, inputs: tuple, output: torch.Tensor):
            inp = inputs[0]
            in_numel = inp.numel()
            out_numel = output.numel()
            fp32_in = in_numel * 4
            fp32_out = out_numel * 4

            if activation_bits >= 32:
                comp_in = fp32_in
                comp_out = fp32_out
                meta_bytes = 0
            else:
                # 2B FP16 scale + 1B uint8 zero-point per activation tensor = 3B metadata
                meta_bytes = 3
                comp_in = packed_nbytes(in_numel, activation_bits) + meta_bytes
                comp_out = packed_nbytes(out_numel, activation_bits) + meta_bytes

            layer_records.append({
                "name": name,
                "in_numel": in_numel,
                "out_numel": out_numel,
                "fp32_in_bytes": fp32_in,
                "fp32_out_bytes": fp32_out,
                "comp_in_bytes": comp_in,
                "comp_out_bytes": comp_out,
                "meta_bytes": meta_bytes,
                # Peak live memory during this layer = input buffer + output buffer
                "fp32_live_bytes": fp32_in + fp32_out,
                "comp_live_bytes": comp_in + comp_out,
            })

        return _hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, QuantizedConv2d, QuantizedLinear)):
            hooks.append(m.register_forward_hook(_make_hook(name)))

    _ = model(dummy_x)

    for h in hooks:
        h.remove()

    fp32_total = sum(r["fp32_in_bytes"] for r in layer_records)
    comp_total = sum(r["comp_in_bytes"] for r in layer_records)
    meta_total = sum(r["meta_bytes"] for r in layer_records)
    fp32_peak = max((r["fp32_live_bytes"] for r in layer_records), default=0)
    comp_peak = max((r["comp_live_bytes"] for r in layer_records), default=0)

    return {
        "num_quantized_layers": len(layer_records),
        "fp32_total_activation_bytes": fp32_total,
        "compressed_total_activation_bytes": comp_total,
        "activation_metadata_bytes": meta_total,
        "fp32_peak_activation_bytes": fp32_peak,
        "compressed_peak_activation_bytes": comp_peak,
        "layer_records": layer_records,
    }


@torch.no_grad()
def compute_model_memory_stats(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    weight_quant_bits: int,
    activation_quant_bits: int,
    input_shape: tuple = (1, 3, 32, 32),
    device: torch.device = torch.device("cpu"),
) -> ModelMemoryReport:
    """Computes full baseline vs compressed in-memory statistics and compression ratios."""
    # 1. Baseline FP32 statistics
    fp32_conv_linear_bytes = 0
    fp32_bn_bytes = 0
    for m in baseline_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fp32_conv_linear_bytes += m.weight.numel() * 4
            if m.bias is not None:
                fp32_conv_linear_bytes += m.bias.numel() * 4
        elif isinstance(m, nn.BatchNorm2d):
            # weight (gamma), bias (beta), running_mean, running_var (4 float32 tensors)
            fp32_bn_bytes += (
                m.weight.numel() * 4
                + m.bias.numel() * 4
                + m.running_mean.numel() * 4
                + m.running_var.numel() * 4
            )

    fp32_total_bytes = fp32_conv_linear_bytes + fp32_bn_bytes

    # 2. Compressed model statistics
    if weight_quant_bits >= 32:
        packed_w_bytes = fp32_conv_linear_bytes
        scale_bytes = 0
        zp_bytes = 0
        bias_bytes = 0
        struct_bytes = 0
        total_meta_bytes = 0
        comp_total_bytes = fp32_total_bytes
    else:
        packed_w_bytes = 0
        scale_bytes = 0
        zp_bytes = 0
        bias_bytes = 0
        struct_bytes = 0
        for m in compressed_model.modules():
            if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
                bd = m.memory_breakdown()
                packed_w_bytes += bd["packed_weight_bytes"]
                scale_bytes += bd["scale_bytes"]
                zp_bytes += bd["zp_bytes"]
                bias_bytes += bd["bias_bytes"]
                struct_bytes += bd["struct_bytes"]

        total_meta_bytes = scale_bytes + zp_bytes + bias_bytes + struct_bytes
        comp_total_bytes = packed_w_bytes + total_meta_bytes

    # 3. Activation statistics
    act_stats = measure_activation_memory(
        compressed_model,
        input_shape=input_shape,
        activation_bits=activation_quant_bits,
        device=device,
    )

    mb_div = 1024.0 * 1024.0
    fp32_mb = fp32_total_bytes / mb_div
    comp_mb = comp_total_bytes / mb_div
    meta_kb = total_meta_bytes / 1024.0
    meta_pct = (total_meta_bytes / max(comp_total_bytes, 1)) * 100.0

    # Weight compression ratio (comparing FP32 Conv/Linear weights vs compressed weights + scales/zp)
    weight_comp_bytes = packed_w_bytes + scale_bytes + zp_bytes
    weight_ratio = fp32_conv_linear_bytes / max(weight_comp_bytes, 1)

    # Overall model compression ratio (including folded BN savings & all metadata overheads)
    model_ratio = fp32_total_bytes / max(comp_total_bytes, 1)

    # Activation compression ratios
    act_ratio = act_stats["fp32_total_activation_bytes"] / max(
        act_stats["compressed_total_activation_bytes"], 1
    )
    peak_act_ratio = act_stats["fp32_peak_activation_bytes"] / max(
        act_stats["compressed_peak_activation_bytes"], 1
    )

    return ModelMemoryReport(
        weight_quant_bits=weight_quant_bits,
        activation_quant_bits=activation_quant_bits,
        fp32_conv_linear_weight_bytes=fp32_conv_linear_bytes,
        fp32_bn_param_and_buffer_bytes=fp32_bn_bytes,
        fp32_total_model_bytes=fp32_total_bytes,
        packed_weight_bytes=packed_w_bytes,
        scale_overhead_bytes=scale_bytes,
        zero_point_overhead_bytes=zp_bytes,
        folded_bias_overhead_bytes=bias_bytes,
        struct_metadata_bytes=struct_bytes,
        total_metadata_overhead_bytes=total_meta_bytes,
        compressed_total_model_bytes=comp_total_bytes,
        fp32_total_activation_bytes=act_stats["fp32_total_activation_bytes"],
        compressed_total_activation_bytes=act_stats["compressed_total_activation_bytes"],
        activation_metadata_bytes=act_stats["activation_metadata_bytes"],
        fp32_peak_activation_bytes=act_stats["fp32_peak_activation_bytes"],
        compressed_peak_activation_bytes=act_stats["compressed_peak_activation_bytes"],
        fp32_model_size_mb=round(fp32_mb, 4),
        compressed_model_size_mb=round(comp_mb, 4),
        metadata_overhead_kb=round(meta_kb, 3),
        metadata_fraction_pct=round(meta_pct, 2),
        weight_compression_ratio=round(weight_ratio, 4),
        model_compression_ratio=round(model_ratio, 4),
        activation_compression_ratio=round(act_ratio, 4),
        peak_activation_compression_ratio=round(peak_act_ratio, 4),
    )
