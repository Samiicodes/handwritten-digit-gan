'''Tests for MNIST loading, pixel scaling, and batching.'''

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from digitgan.data import (
    DigitImages,
    balanced_labels,
    load_mnist,
    make_loader,
    scale_pixels,
    to_unit_range,
)


def write_idx_files(raw_dir: Path, prefix: str, pixels: np.ndarray, labels: np.ndarray) -> None:
    '''Write images and labels in the IDX format that MNIST is distributed in.'''
    raw_dir.mkdir(parents=True, exist_ok=True)
    count, height, width = pixels.shape
    image_header = struct.pack('>IIII', 2051, count, height, width)
    (raw_dir / f'{prefix}-images-idx3-ubyte').write_bytes(image_header + pixels.tobytes())
    label_header = struct.pack('>II', 2049, count)
    (raw_dir / f'{prefix}-labels-idx1-ubyte').write_bytes(label_header + labels.tobytes())


def test_scaling_maps_pixels_to_the_tanh_range() -> None:
    pixels = torch.tensor([0, 51, 255], dtype=torch.uint8)

    scaled = scale_pixels(pixels)

    assert scaled.tolist() == pytest.approx([-1.0, -0.6, 1.0])
    assert to_unit_range(scaled).tolist() == pytest.approx([0.0, 0.2, 1.0])


def test_load_mnist_reads_idx_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / 'MNIST' / 'raw'
    for prefix, count in (('train', 6), ('t10k', 4)):
        pixels = rng.integers(0, 256, size=(count, 28, 28), dtype=np.uint8)
        labels = np.arange(count, dtype=np.uint8)
        write_idx_files(raw_dir, prefix, pixels, labels)

    data = load_mnist(tmp_path, 'test', download=False)

    assert data.images.shape == (4, 1, 28, 28)
    assert data.images.dtype == torch.float32
    assert data.images.min() >= -1
    assert data.images.max() <= 1
    assert data.labels.tolist() == [0, 1, 2, 3]


def test_digit_images_reject_wrong_shapes() -> None:
    with pytest.raises(ValueError, match='shape'):
        DigitImages(torch.zeros(2, 28, 28), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match='length'):
        DigitImages(torch.zeros(2, 1, 28, 28), torch.zeros(3, dtype=torch.long))


def test_head_keeps_the_first_examples(train_data: DigitImages) -> None:
    head = train_data.head(5)

    assert len(head) == 5
    assert torch.equal(head.labels, train_data.labels[:5])


def test_loader_order_depends_only_on_the_seed(train_data: DigitImages) -> None:
    def first_labels(seed: int) -> list[int]:
        _, labels = next(iter(make_loader(train_data, batch_size=20, seed=seed)))
        return [int(label) for label in labels]

    assert first_labels(3) == first_labels(3)
    assert first_labels(3) != first_labels(4)


def test_balanced_labels_cycle_through_every_digit() -> None:
    labels = balanced_labels(30)

    assert torch.bincount(labels).tolist() == [3] * 10
