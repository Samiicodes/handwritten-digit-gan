'''Tests for sampling and for turning samples into pictures.'''

import numpy as np
import pytest
import torch

from digitgan.checkpoints import GeneratorCheckpoint
from digitgan.generation import (
    generate_images,
    generate_number,
    grid_inputs,
    interpolate_styles,
    png_bytes,
    render_grid,
    sample_noise,
    tile,
    to_uint8,
)
from digitgan.models import build_generator


def test_noise_is_reproducible_and_leaves_the_global_rng_alone() -> None:
    torch.manual_seed(0)
    expected_next = torch.rand(1)
    torch.manual_seed(0)

    first = sample_noise(4, 8, seed=5)

    assert torch.equal(first, sample_noise(4, 8, seed=5))
    assert not torch.equal(first, sample_noise(4, 8, seed=6))
    assert torch.equal(torch.rand(1), expected_next)


def test_generate_images_batches_and_restores_training_mode(
    conditional_checkpoint: GeneratorCheckpoint,
) -> None:
    generator = conditional_checkpoint.generator.train()
    noise = sample_noise(5, generator.latent_dim, seed=0)
    labels = torch.arange(5)

    batched = generate_images(generator, noise, labels, batch_size=2)

    assert batched.shape == (5, 1, 28, 28)
    # Batch size changes the convolution kernels' summation order, not the result
    whole = generate_images(generator, noise, labels, batch_size=5)
    assert torch.allclose(batched, whole, atol=1e-5)
    assert generator.training


def test_conditional_grid_shares_noise_across_each_row(
    conditional_checkpoint: GeneratorCheckpoint,
) -> None:
    noise, labels = grid_inputs(conditional_checkpoint.generator, seed=0, rows=3)

    assert labels is not None
    assert labels.tolist() == list(range(10)) * 3
    assert torch.equal(noise[0], noise[9])
    assert not torch.equal(noise[9], noise[10])


def test_unconditional_grid_uses_independent_noise() -> None:
    generator = build_generator('fully_connected', conditional=False, latent_dim=8)

    noise, labels = grid_inputs(generator, seed=0)

    assert labels is None
    assert len(torch.unique(noise, dim=0)) == 100


def test_generate_number_is_reproducible(conditional_checkpoint: GeneratorCheckpoint) -> None:
    generator = conditional_checkpoint.generator

    images = generate_number(generator, [2, 0, 2, 6], variations=3, seed=11)

    assert images.shape == (3, 4, 1, 28, 28)
    assert torch.equal(images, generate_number(generator, [2, 0, 2, 6], variations=3, seed=11))
    # Repeated digits in one row come from the same noise, so they are identical
    assert torch.equal(images[0, 0], images[0, 2])


def test_writing_digits_needs_a_conditional_generator() -> None:
    generator = build_generator('convolutional', conditional=False, latent_dim=8)

    with pytest.raises(ValueError, match='conditional'):
        generate_number(generator, [1], variations=1, seed=0)
    with pytest.raises(ValueError, match='conditional'):
        interpolate_styles(generator, seed=0, steps=3)


def test_interpolation_starts_and_ends_at_the_two_noise_vectors(
    conditional_checkpoint: GeneratorCheckpoint,
) -> None:
    generator = conditional_checkpoint.generator

    steps = interpolate_styles(generator, seed=4, steps=5)

    start, end = sample_noise(2, generator.latent_dim, seed=4)
    first_seven = generate_images(generator, start.unsqueeze(0), torch.tensor([7]))
    last_seven = generate_images(generator, end.unsqueeze(0), torch.tensor([7]))
    assert steps.shape == (10, 5, 1, 28, 28)
    assert torch.allclose(steps[7, 0], first_seven[0], atol=1e-5)
    assert torch.allclose(steps[7, -1], last_seven[0], atol=1e-5)


def test_to_uint8_maps_the_tanh_range_to_bytes() -> None:
    pixels = to_uint8(torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3))

    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[[0, 128, 255]]]


def test_tile_places_images_with_padding() -> None:
    images = np.arange(1, 5, dtype=np.uint8).reshape(2, 2, 1, 1)

    canvas = tile(images, padding=1, fill=9)

    assert canvas.tolist() == [
        [9, 9, 9, 9, 9],
        [9, 1, 9, 2, 9],
        [9, 9, 9, 9, 9],
        [9, 3, 9, 4, 9],
        [9, 9, 9, 9, 9],
    ]


def test_render_grid_scales_and_encodes_png() -> None:
    images = torch.zeros(2, 3, 1, 28, 28)

    picture = render_grid(images, scale=2, padding=0)

    assert picture.size == (3 * 28 * 2, 2 * 28 * 2)
    assert png_bytes(picture).startswith(b'\x89PNG')
