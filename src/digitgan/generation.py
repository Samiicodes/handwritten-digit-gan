'''Sampling from trained generators and turning samples into pictures.'''

import io
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from digitgan.data import IMAGE_SIZE, NUM_CLASSES, to_unit_range
from digitgan.models import Generator


def sample_noise(count: int, latent_dim: int, seed: int) -> Tensor:
    '''Draw standard normal noise from a private generator, leaving the global RNG untouched.'''
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(count, latent_dim, generator=rng)


@torch.no_grad()
def generate_images(
    generator: Generator, noise: Tensor, labels: Tensor | None = None, batch_size: int = 1000
) -> Tensor:
    '''Run the generator in evaluation mode and return CPU images in [-1, 1].

    Args:
        generator: Trained or untrained generator on any device.
        noise: Noise vectors, one per image.
        labels: Requested digits, one per image. Required for conditional generators only.
        batch_size: Images generated per forward pass.

    Returns:
        Images with shape (n, 1, 28, 28).
    '''
    device = next(generator.parameters()).device
    was_training = generator.training
    # Batch norm must use its running statistics, or each image would depend on its batch
    generator.eval()
    try:
        batches = []
        for start in range(0, len(noise), batch_size):
            batch_noise = noise[start : start + batch_size].to(device)
            batch_labels = None if labels is None else labels[start : start + batch_size].to(device)
            images: Tensor = generator(batch_noise, batch_labels)
            batches.append(images.cpu())
    finally:
        generator.train(was_training)
    return torch.cat(batches)


def grid_inputs(
    generator: Generator, seed: int, rows: int = NUM_CLASSES
) -> tuple[Tensor, Tensor | None]:
    '''Return noise and labels for a sample grid with ten columns.

    A conditional generator gets one noise vector per row and the digits 0 to 9 across the
    columns, so each row shows one handwriting style. An unconditional generator gets
    independent noise for every cell.
    '''
    if generator.conditional:
        noise = sample_noise(rows, generator.latent_dim, seed).repeat_interleave(NUM_CLASSES, dim=0)
        return noise, torch.arange(NUM_CLASSES).repeat(rows)
    return sample_noise(rows * NUM_CLASSES, generator.latent_dim, seed), None


def generate_number(
    generator: Generator, digits: Sequence[int], variations: int, seed: int
) -> Tensor:
    '''Write a sequence of digits once per variation.

    Every digit in a row shares one noise vector, which keeps the handwriting style of that
    row consistent.

    Returns:
        Images with shape (variations, len(digits), 1, 28, 28).
    '''
    if not generator.conditional:
        raise ValueError('writing chosen digits needs a conditional generator')
    noise = sample_noise(variations, generator.latent_dim, seed).repeat_interleave(
        len(digits), dim=0
    )
    labels = torch.tensor(list(digits), dtype=torch.long).repeat(variations)
    images = generate_images(generator, noise, labels)
    return images.view(variations, len(digits), 1, IMAGE_SIZE, IMAGE_SIZE)


def interpolate_styles(generator: Generator, seed: int, steps: int) -> Tensor:
    '''Walk in a straight line between two noise vectors while keeping each digit fixed.

    Returns:
        Images with shape (10, steps, 1, 28, 28): one row per digit.
    '''
    if not generator.conditional:
        raise ValueError('interpolating per digit needs a conditional generator')
    start, end = sample_noise(2, generator.latent_dim, seed)
    weights = torch.linspace(0, 1, steps).unsqueeze(1)
    path = (1 - weights) * start + weights * end
    noise = path.repeat(NUM_CLASSES, 1)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(steps)
    images = generate_images(generator, noise, labels)
    return images.view(NUM_CLASSES, steps, 1, IMAGE_SIZE, IMAGE_SIZE)


def to_uint8(images: Tensor) -> np.ndarray:
    '''Convert images in [-1, 1] of shape (..., 1, 28, 28) to 8-bit arrays of (..., 28, 28).'''
    pixels = to_unit_range(images).squeeze(-3).mul(255).round().to(torch.uint8)
    return pixels.numpy()


def tile(images: np.ndarray, padding: int = 2, fill: int = 0) -> np.ndarray:
    '''Arrange an array of shape (rows, columns, height, width) into one picture.

    Args:
        images: 8-bit images.
        padding: Pixels between neighbouring images and around the edge.
        fill: Grey level of the padding.
    '''
    rows, columns, height, width = images.shape
    canvas = np.full(
        (rows * (height + padding) + padding, columns * (width + padding) + padding),
        fill,
        dtype=images.dtype,
    )
    for row in range(rows):
        for column in range(columns):
            top = padding + row * (height + padding)
            left = padding + column * (width + padding)
            canvas[top : top + height, left : left + width] = images[row, column]
    return canvas


def render_grid(images: Tensor, scale: int = 1, padding: int = 0) -> Image.Image:
    '''Render images of shape (rows, columns, 1, 28, 28) as a greyscale picture.

    Without padding the digits of a row join up like a written number. MNIST digits already
    have a blank border, so they do not touch.
    '''
    canvas = Image.fromarray(tile(to_uint8(images), padding), mode='L')
    if scale == 1:
        return canvas
    # Nearest neighbour keeps the pixels sharp instead of blurring them
    return canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)


def png_bytes(picture: Image.Image) -> bytes:
    '''Encode a picture as PNG.'''
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    return buffer.getvalue()
