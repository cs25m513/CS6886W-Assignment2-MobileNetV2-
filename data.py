"""CIFAR-10 dataset preparation, normalization, and augmentation pipeline (Question 1a)."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from config import CIFAR10_MEAN, CIFAR10_STD


class Cutout:
    """Randomly masks out one or more square patches from a tensor image.

    Reference: DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks
    with Cutout", arXiv:1708.04552. Implemented from scratch without external libraries.
    """

    def __init__(self, n_holes: int = 1, length: int = 8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Applies Cutout mask to a normalized tensor image of shape (C, H, W)."""
        _, h, w = img.shape
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        return img * mask.unsqueeze(0)


def get_cifar10_transforms() -> Tuple[T.Compose, T.Compose]:
    """Returns training and test transforms for CIFAR-10 (32x32 RGB images).

    Training Pipeline:
      1. RandomCrop(32, padding=4, padding_mode='reflect')
      2. RandomHorizontalFlip(p=0.5)
      3. ToTensor() -> scales [0, 255] uint8 to [0.0, 1.0] float32
      4. Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
      5. Cutout(n_holes=1, length=8)

    Evaluation Pipeline:
      1. ToTensor()
      2. Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        Cutout(n_holes=1, length=8),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    calib_samples: int = 512,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates CIFAR-10 train, calibration (unaugmented train subset), and test DataLoaders."""
    train_transform, test_transform = get_cifar10_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    # Calibration dataset uses test_transform (no random crop/cutout) on training images
    calib_full_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(calib_full_dataset), generator=g)[:calib_samples].tolist()
    calib_dataset = Subset(calib_full_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, calib_loader, test_loader
