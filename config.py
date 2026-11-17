"""Global configuration, reproducibility seed setup, and CIFAR-10 constants."""

import os
import random
from dataclasses import dataclass
import numpy as np
import torch


# CIFAR-10 channel-wise mean and std computed on the 50,000 training images
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass
class TrainConfig:
    """Configuration for baseline MobileNet-v2 training on CIFAR-10."""

    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    seed: int = 42

    # Architecture settings
    num_classes: int = 10
    width_mult: float = 1.0
    dropout: float = 0.2

    # Optimization settings
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    num_workers: int = 4


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
