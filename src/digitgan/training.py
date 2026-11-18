'''Adversarial training loop shared by every GAN in the comparison.'''

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from digitgan.checkpoints import (
    GENERATOR_FILE,
    PROGRESS_FILE,
    EpochStats,
    GeneratorCheckpoint,
    run_name,
    save_checkpoint,
)
from digitgan.config import GanConfig, TrainingConfig
from digitgan.data import DigitImages, make_loader
from digitgan.generation import generate_images, grid_inputs, to_uint8
from digitgan.models import Discriminator, Generator, build_discriminator, build_generator

logger = logging.getLogger(__name__)

# Maps a generator to a quality score, such as FID, measured during training
QualityTracker = Callable[[Generator], float]


@dataclass(frozen=True)
class TrainingRun:
    '''A trained generator, its training record, and its progress snapshots.'''

    checkpoint: GeneratorCheckpoint
    # 8-bit sample grids with shape (epochs + 1, 100, 28, 28), starting before training
    progress: np.ndarray


@dataclass
class _EpochTotals:
    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    real_score: float = 0.0
    fake_score: float = 0.0
    batches: int = 0


def train_gan(
    model_name: str,
    model_config: GanConfig,
    training: TrainingConfig,
    train_data: DigitImages,
    seed: int,
    device: torch.device,
    tracker: QualityTracker | None = None,
) -> TrainingRun:
    '''Train one GAN from scratch.

    Args:
        model_name: Key of the model in the configuration.
        model_config: Architecture and conditioning of the model.
        training: Settings shared by every model.
        train_data: Training images and labels.
        seed: Seed for weight initialisation, noise, dropout, and batch order.
        device: Device to train on.
        tracker: Optional quality measure, called every tracking_interval epochs.

    Returns:
        The trained generator with its history and progress snapshots.
    '''
    torch.manual_seed(seed)
    generator = build_generator(
        model_config.architecture, model_config.conditional, training.latent_dim
    ).to(device)
    discriminator = build_discriminator(model_config.architecture, model_config.conditional).to(
        device
    )
    generator_optimiser = Adam(
        generator.parameters(), lr=training.learning_rate, betas=training.adam_betas
    )
    discriminator_optimiser = Adam(
        discriminator.parameters(), lr=training.learning_rate, betas=training.adam_betas
    )
    loader = make_loader(train_data, training.batch_size, seed)
    progress_noise, progress_labels = grid_inputs(generator, seed)

    frames = [to_uint8(generate_images(generator, progress_noise, progress_labels))]
    history: list[EpochStats] = []
    for epoch in range(1, training.epochs + 1):
        started = time.perf_counter()
        totals = _train_epoch(
            generator,
            discriminator,
            loader,
            generator_optimiser,
            discriminator_optimiser,
            training.real_label_target,
            device,
        )
        tracking_epoch = epoch % training.tracking_interval == 0 or epoch == training.epochs
        fid = tracker(generator) if tracker is not None and tracking_epoch else None
        stats = EpochStats(
            epoch=epoch,
            generator_loss=totals.generator_loss / totals.batches,
            discriminator_loss=totals.discriminator_loss / totals.batches,
            real_score=totals.real_score / totals.batches,
            fake_score=totals.fake_score / totals.batches,
            seconds=time.perf_counter() - started,
            fid=fid,
        )
        history.append(stats)
        frames.append(to_uint8(generate_images(generator, progress_noise, progress_labels)))
        if tracking_epoch:
            _log_epoch(stats, training.epochs)

    checkpoint = GeneratorCheckpoint(
        model_name=model_name,
        display_name=model_config.display_name,
        architecture=model_config.architecture,
        conditional=model_config.conditional,
        seed=seed,
        generator=generator.eval(),
        history=tuple(history),
    )
    return TrainingRun(checkpoint=checkpoint, progress=np.stack(frames))


def _train_epoch(
    generator: Generator,
    discriminator: Discriminator,
    loader: torch.utils.data.DataLoader[tuple[Tensor, ...]],
    generator_optimiser: Adam,
    discriminator_optimiser: Adam,
    real_label_target: float,
    device: torch.device,
) -> _EpochTotals:
    generator.train()
    discriminator.train()
    loss_function = nn.BCEWithLogitsLoss()
    totals = _EpochTotals()

    for real_images, digits in loader:
        real_images = real_images.to(device)
        labels = digits.to(device) if generator.conditional else None
        batch_size = real_images.size(0)
        real_targets = torch.full((batch_size,), real_label_target, device=device)
        fake_targets = torch.zeros(batch_size, device=device)

        # Discriminator step. Detach the fakes so this update does not reach the generator.
        noise = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_images = generator(noise, labels)
        real_logits = discriminator(real_images, labels)
        fake_logits = discriminator(fake_images.detach(), labels)
        discriminator_loss = loss_function(real_logits, real_targets) + loss_function(
            fake_logits, fake_targets
        )
        discriminator_optimiser.zero_grad()
        discriminator_loss.backward()
        discriminator_optimiser.step()

        # Generator step with fresh noise and the non-saturating loss. Smoothing is one-sided,
        # so the generator still aims for a full 1.0.
        noise = torch.randn(batch_size, generator.latent_dim, device=device)
        generated_logits = discriminator(generator(noise, labels), labels)
        generator_loss = loss_function(generated_logits, torch.ones_like(generated_logits))
        generator_optimiser.zero_grad()
        generator_loss.backward()
        generator_optimiser.step()

        totals.generator_loss += generator_loss.item()
        totals.discriminator_loss += discriminator_loss.item()
        totals.real_score += torch.sigmoid(real_logits).mean().item()
        totals.fake_score += torch.sigmoid(fake_logits).mean().item()
        totals.batches += 1
    return totals


def _log_epoch(stats: EpochStats, epochs: int) -> None:
    message = (
        f'Epoch {stats.epoch}/{epochs}: generator loss {stats.generator_loss:.3f}, '
        f'discriminator loss {stats.discriminator_loss:.3f}'
    )
    if stats.fid is not None:
        message += f', FID {stats.fid:.2f}'
    logger.info(message)


def save_run(run: TrainingRun, run_dir: Path) -> Path:
    '''Write the checkpoint and progress snapshots of one run, and return its directory.'''
    checkpoint = run.checkpoint
    directory = run_dir / run_name(checkpoint.model_name, checkpoint.seed)
    save_checkpoint(checkpoint, directory / GENERATOR_FILE)
    np.savez_compressed(directory / PROGRESS_FILE, frames=run.progress)
    return directory
