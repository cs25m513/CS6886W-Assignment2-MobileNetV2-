"""Compression Sweep & WandB Parallel Coordinates Logger (Question 3).

Runs >= 12 simulation configurations across weight & activation bit-widths:
(W8A8, W8A6, W6A8, W6A6, W6A4, W4A8, W4A6, W4A4, W3A6, W3A4, W3A3, W2A4, W2A2)
Logs each run to Weights & Biases (`wandb`) and exports `./results/compression_sweep.csv`.

Usage:
    python sweep.py --checkpoint ./checkpoints/mobilenetv2_cifar10_best.pth --use_wandb
"""

import argparse
import csv
import os
from typing import List, Tuple
import torch

from config import set_seed
from data import get_dataloaders
from models import mobilenet_v2_cifar10
from compression import compress_mobilenetv2, compute_model_memory_stats
from test import evaluate_accuracy


# 13 simulation runs (exceeds the minimum of 8 runs required by Question 3b)
SWEEP_CONFIGS: List[Tuple[int, int]] = [
    (8, 8),
    (8, 6),
    (6, 8),
    (6, 6),
    (6, 4),
    (4, 8),
    (4, 6),
    (4, 4),
    (3, 6),
    (3, 4),
    (3, 3),
    (2, 4),
    (2, 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MobileNet-v2 Compression Sweep (Q3)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/mobilenetv2_cifar10_best.pth",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_percentile", type=float, default=99.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs6886w-assignment2-sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
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
        print(f"[Sweep] Loaded baseline checkpoint: {args.checkpoint}")
    else:
        print(f"[Sweep] Warning: checkpoint {args.checkpoint} not found.")

    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device)
    print(f"[Sweep] Baseline FP32 Accuracy: {baseline_acc:.2f}%\n")

    csv_path = os.path.join(args.results_dir, "compression_sweep.csv")
    fieldnames = [
        "run_id",
        "weight_quant_bits",
        "activation_quant_bits",
        "baseline_acc",
        "quantized_acc",
        "acc_drop",
        "fp32_model_size_mb",
        "model_size_mb",
        "packed_weight_mb",
        "metadata_overhead_kb",
        "weight_compression_ratio",
        "compression_ratio",
        "activation_compression_ratio",
        "peak_activation_compression_ratio",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (w_bits, a_bits) in enumerate(SWEEP_CONFIGS, start=1):
            run_name = f"W{w_bits}A{a_bits}"
            compressed_model = compress_mobilenetv2(
                baseline_model,
                weight_quant_bits=w_bits,
                activation_quant_bits=a_bits,
                fold_bn=True,
                clip_percentile=args.clip_percentile,
            ).to(device)

            q_acc = evaluate_accuracy(compressed_model, test_loader, device)
            mem = compute_model_memory_stats(
                baseline_model=baseline_model,
                compressed_model=compressed_model,
                weight_quant_bits=w_bits,
                activation_quant_bits=a_bits,
                input_shape=(1, 3, 32, 32),
                device=device,
            )

            row = {
                "run_id": run_name,
                "weight_quant_bits": w_bits,
                "activation_quant_bits": a_bits,
                "baseline_acc": round(baseline_acc, 2),
                "quantized_acc": round(q_acc, 2),
                "acc_drop": round(baseline_acc - q_acc, 2),
                "fp32_model_size_mb": mem.fp32_model_size_mb,
                "model_size_mb": mem.compressed_model_size_mb,
                "packed_weight_mb": round(mem.packed_weight_bytes / (1024.0 ** 2), 4),
                "metadata_overhead_kb": mem.metadata_overhead_kb,
                "weight_compression_ratio": mem.weight_compression_ratio,
                "compression_ratio": mem.model_compression_ratio,
                "activation_compression_ratio": mem.activation_compression_ratio,
                "peak_activation_compression_ratio": mem.peak_activation_compression_ratio,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"[{idx:02d}/{len(SWEEP_CONFIGS)}] {run_name:<6} | "
                f"Acc: {q_acc:6.2f}% (Drop: {baseline_acc - q_acc:+5.2f}%) | "
                f"Size: {mem.compressed_model_size_mb:6.3f} MB | "
                f"Model Ratio: {mem.model_compression_ratio:5.2f}x | "
                f"Weight Ratio: {mem.weight_compression_ratio:5.2f}x | "
                f"Act Ratio: {mem.activation_compression_ratio:5.2f}x"
            )

            if args.use_wandb:
                import wandb
                run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={
                        "activation_quant_bits": a_bits,
                        "weight_quant_bits": w_bits,
                    },
                    reinit=True,
                )
                wandb.log({
                    "activation_quant_bits": a_bits,
                    "weight_quant_bits": w_bits,
                    "compression_ratio": mem.model_compression_ratio,
                    "weight_compression_ratio": mem.weight_compression_ratio,
                    "activation_compression_ratio": mem.activation_compression_ratio,
                    "model_size_mb": mem.compressed_model_size_mb,
                    "quantized_acc": q_acc,
                })
                run.finish()

    print(f"\n[Sweep Complete] Saved {len(SWEEP_CONFIGS)} runs to {csv_path}")


if __name__ == "__main__":
    main()
