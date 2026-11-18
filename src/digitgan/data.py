'''MNIST loading and the pixel scaling shared by every model.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

IMAGE_SIZE = 28
NUM_CLASSES = 10

Split = Literal['train', 'test']


@dataclass(frozen=True)
class DigitImages:
    '''Images scaled to [-1, 1] with shape (n, 1, 28, 28), and their digit labels.'''

    images: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        if self.images.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f'expected images of shape (n, 1, 28, 28), got {self.images.shape}')
        if len(self.images) != len(self.labels):
            raise ValueError('images and labels differ in length')

    def __len__(self) -> int:
        return len(self.labels)

    def head(self, size: int) -> 'DigitImages':
        '''Return the first size examples, keeping the original order.'''
        return DigitImages(self.images[:size], self.labels[:size])


def scale_pixels(pixels: Tensor) -> Tensor:
    '''Map 8-bit pixels in [0, 255] to [-1, 1], the range of the generators' tanh output.'''
    return pixels.float().div(255).mul(2).sub(1)


def to_unit_range(images: Tensor) -> Tensor:
    '''Map images from [-1, 1] back to [0, 1] for display.'''
    return images.add(1).div(2).clamp(0, 1)


def load_mnist(root: Path, split: Split, download: bool = True) -> DigitImages:
    '''Load one MNIST split into memory.

    The whole split fits in a few hundred megabytes, and holding it as one tensor is much
    faster on CPU than transforming images one at a time in a data loader.

    Args:
        root: Directory that holds, or will hold, the MNIST folder.
        split: The fixed 60,000-image training split or the 10,000-image test split.
        download: Download the files when they are missing. torchvision checks their MD5 sums.

    Returns:
        The scaled images and their labels.
    '''
    dataset = MNIST(root=root, train=split == 'train', download=download)
    return DigitImages(scale_pixels(dataset.data).unsqueeze(1), dataset.targets.clone())


def make_loader(data: DigitImages, batch_size: int, seed: int) -> DataLoader[tuple[Tensor, ...]]:
    '''Return a shuffling loader whose order depends only on the seed.'''
    shuffle_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(data.images, data.labels),
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )


def balanced_labels(count: int) -> Tensor:
    '''Return count labels cycling through the digits 0 to 9, so each digit appears equally.'''
    return torch.arange(count) % NUM_CLASSES
