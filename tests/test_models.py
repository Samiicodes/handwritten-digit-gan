'''Tests for the generator and discriminator architectures.'''

import pytest
import torch

from digitgan.config import Architecture
from digitgan.models import (
    Discriminator,
    Generator,
    build_discriminator,
    build_generator,
    count_parameters,
)

ARCHITECTURES: list[Architecture] = ['fully_connected', 'convolutional']


@pytest.mark.parametrize('architecture', ARCHITECTURES)
@pytest.mark.parametrize('conditional', [False, True])
def test_networks_produce_images_and_logits(architecture: Architecture, conditional: bool) -> None:
    generator = build_generator(architecture, conditional, latent_dim=16)
    discriminator = build_discriminator(architecture, conditional)
    labels = torch.arange(4) if conditional else None

    images = generator(torch.randn(4, 16), labels)
    logits = discriminator(images, labels)

    assert images.shape == (4, 1, 28, 28)
    assert images.min() >= -1
    assert images.max() <= 1
    assert logits.shape == (4,)


def test_convolutional_generator_outputs_full_size_images() -> None:
    # The coursework DCGAN produced 24x24 images, which the discriminator then had to pool
    generator = build_generator('convolutional', conditional=False, latent_dim=100)

    assert generator(torch.randn(2, 100)).shape[-2:] == (28, 28)


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_conditional_networks_need_labels(architecture: Architecture) -> None:
    generator = build_generator(architecture, conditional=True, latent_dim=8)
    discriminator = build_discriminator(architecture, conditional=True)

    with pytest.raises(ValueError, match='needs labels'):
        generator(torch.randn(2, 8))
    with pytest.raises(ValueError, match='needs labels'):
        discriminator(torch.zeros(2, 1, 28, 28))


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_unconditional_networks_refuse_labels(architecture: Architecture) -> None:
    generator = build_generator(architecture, conditional=False, latent_dim=8)
    discriminator = build_discriminator(architecture, conditional=False)
    labels = torch.tensor([1, 2])

    with pytest.raises(ValueError, match='unconditional'):
        generator(torch.randn(2, 8), labels)
    with pytest.raises(ValueError, match='unconditional'):
        discriminator(torch.zeros(2, 1, 28, 28), labels)


def test_the_label_changes_the_generated_image() -> None:
    torch.manual_seed(0)
    generator = build_generator('convolutional', conditional=True, latent_dim=8).eval()
    noise = torch.randn(1, 8)

    first = generator(noise, torch.tensor([3]))
    second = generator(noise, torch.tensor([7]))

    assert not torch.allclose(first, second)


def test_conditioning_adds_only_the_label_embeddings() -> None:
    unconditional = build_generator('convolutional', conditional=False, latent_dim=100)
    conditional = build_generator('convolutional', conditional=True, latent_dim=100)
    # Ten embedding vectors of size ten, and ten more inputs to the first linear layer
    extra = 10 * 10 + 10 * 128 * 7 * 7

    assert count_parameters(conditional) - count_parameters(unconditional) == extra


def test_base_classes_must_be_subclassed() -> None:
    with pytest.raises(NotImplementedError):
        Generator(latent_dim=8, conditional=False)(torch.randn(1, 8))
    with pytest.raises(NotImplementedError):
        Discriminator(conditional=False)(torch.zeros(1, 1, 28, 28))
