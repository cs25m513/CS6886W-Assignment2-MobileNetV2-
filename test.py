"""Evaluation & Compression CLI for MobileNet-v2 on CIFAR-10 (Questions 2, 3, 4).

Matches the exact template repository interface specified in the assignment:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8
"""

import argparse
import json
import os
import torch
import torch.nn as nn

from config import set_seed
from data import get_dataloaders
from models import mobilenet_v2_cifar10
from compression import (
    compress_mobilenetv2,
    compute_model_memory_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate & Compress MobileNet-v2 on CIFAR-10 (CS6886W Assignment 2)"
    )
    parser.add_argument(
        "--weight_quant_bits",
        type=int,
        default=8,
        help="Bit-width for weight quantization (2..8, or 32 for uncompressed FP32).",
    )
    parser.add_argument(
        "--activation_quant_bits",
        type=int,
        default=8,
        help="Bit-width for activation quantization (2..8, or 32 for uncompressed FP32).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mobilenetv2_cifar10_best.pth",
        help="Path to pretrained MobileNet-v2 checkpoint.",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_percentile", type=float, default=99.95)
    parser.add_argument("--pack_activations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    baseline_model = mobilenet_v2_cifar10(num_classes=10).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        baseline_model.load_state_dict(state_dict)
        print(f"[Loaded Checkpoint] {args.checkpoint}")
    else:
        print(
            f"[Warning] Checkpoint '{args.checkpoint}' not found; running on initialized weights."
        )

    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device)

    compressed_model = compress_mobilenetv2(
        baseline_model,
        weight_quant_bits=args.weight_quant_bits,
        activation_quant_bits=args.activation_quant_bits,
        fold_bn=True,
        clip_percentile=args.clip_percentile,
        pack_activations=args.pack_activations,
    ).to(device)

    quantized_acc = evaluate_accuracy(compressed_model, test_loader, device)

    mem_report = compute_model_memory_stats(
        baseline_model=baseline_model,
        compressed_model=compressed_model,
        weight_quant_bits=args.weight_quant_bits,
        activation_quant_bits=args.activation_quant_bits,
        input_shape=(1, 3, 32, 32),
        device=device,
    )

    print("=" * 74)
    print(
        f"CS6886W Assignment 2 | MobileNet-v2 Compression Report "
        f"(W{args.weight_quant_bits}A{args.activation_quant_bits})"
    )
    print("=" * 74)
    print(f"Baseline Uncompressed Top-1 Accuracy      : {baseline_acc:.2f}%")
    print(f"Compressed (W{args.weight_quant_bits}A{args.activation_quant_bits}) Top-1 Accuracy          : {quantized_acc:.2f}%")
    print(f"Accuracy Drop                             : {baseline_acc - quantized_acc:+.2f}%")
    print("-" * 74)
    print(
        f"Baseline Model In-Memory Size (FP32)      : {mem_report.fp32_model_size_mb:.4f} MB "
        f"({mem_report.fp32_total_model_bytes:,} B)"
    )
    print(
        f"Compressed Packed Weights Size            : "
        f"{mem_report.packed_weight_bytes / (1024**2):.4f} MB "
        f"({mem_report.packed_weight_bytes:,} B)"
    )
    print(
        f"Storage Overheads (Scales+ZP+FoldedBias)  : {mem_report.metadata_overhead_kb:.2f} KB "
        f"({mem_report.total_metadata_overhead_bytes:,} B | {mem_report.metadata_fraction_pct:.2f}%)"
    )
    print(
        f"Final Compressed Model Size (In-Memory)   : {mem_report.compressed_model_size_mb:.4f} MB "
        f"({mem_report.compressed_total_model_bytes:,} B)"
    )
    print("-" * 74)
    print(f"Weight Compression Ratio (In-Memory)      : {mem_report.weight_compression_ratio:.2f}x")
    print(f"Overall Model Compression Ratio (In-Mem)  : {mem_report.model_compression_ratio:.2f}x")
    print(
        f"Activation Compression Ratio (Total)      : {mem_report.activation_compression_ratio:.2f}x "
        f"({mem_report.fp32_total_activation_bytes / 1024:.1f} KB -> "
        f"{mem_report.compressed_total_activation_bytes / 1024:.1f} KB)"
    )
    print(
        f"Activation Compression Ratio (Peak Live)  : {mem_report.peak_activation_compression_ratio:.2f}x "
        f"({mem_report.fp32_peak_activation_bytes / 1024:.1f} KB -> "
        f"{mem_report.compressed_peak_activation_bytes / 1024:.1f} KB)"
    )
    print("=" * 74)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        payload = mem_report.to_dict()
        payload["baseline_acc"] = round(baseline_acc, 2)
        payload["quantized_acc"] = round(quantized_acc, 2)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
