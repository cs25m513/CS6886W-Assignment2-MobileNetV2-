"""Training script for baseline MobileNet-v2 on CIFAR-10 (Question 1).

Usage:
    python train.py --epochs 200 --batch_size 128 --lr 0.05 --seed 42
"""

import argparse
import csv
import os
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig, set_seed
from data import get_dataloaders
from models import mobilenet_v2_cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 MobileNet-v2 Baseline (CS6886W Assignment 2)"
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true", help="Log training metrics to WandB")
    parser.add_argument("--wandb_project", type=str, default="cs6886w-assignment2")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device} | Seed: {args.seed}")

    train_loader, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = mobilenet_v2_cifar10(
        num_classes=10, width_mult=args.width_mult, dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] MobileNetV2 (width={args.width_mult}) | Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # Linear warmup followed by Cosine Annealing LR schedule
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="mobilenetv2-cifar10-baseline", config=vars(args))

    history_path = os.path.join(args.results_dir, "training_history.csv")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "mobilenetv2_cifar10_best.pth")
    best_acc = 0.0

    with open(history_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time_s"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, eval_criterion, device)
            scheduler.step()
            dt = time.time() - t0

            writer.writerow([
                epoch,
                f"{current_lr:.6f}",
                f"{train_loss:.4f}",
                f"{train_acc:.2f}",
                f"{test_loss:.4f}",
                f"{test_acc:.2f}",
                f"{dt:.2f}",
            ])
            f.flush()

            if args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                })

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_acc": best_acc,
                        "config": vars(args),
                    },
                    best_ckpt_path,
                )

            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"LR: {current_lr:.5f} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)"
                )

    print(f"\n[Done] Best Test Top-1 Accuracy: {best_acc:.2f}% saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
