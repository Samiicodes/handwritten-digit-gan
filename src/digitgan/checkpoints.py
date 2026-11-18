'''Saving and loading trained generators together with their training record.'''

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from digitgan import __version__
from digitgan.config import Architecture, GanConfig
from digitgan.models import Generator, build_generator

CHECKPOINT_FORMAT = 1
GENERATOR_FILE = 'generator.pt'
PROGRESS_FILE = 'progress.npz'


@dataclass(frozen=True)
class EpochStats:
    '''Averages over one training epoch.'''

    epoch: int
    generator_loss: float
    discriminator_loss: float
    # Mean probability the discriminator gives to real and to generated images being real
    real_score: float
    fake_score: float
    seconds: float
    # Only recorded on tracking epochs
    fid: float | None = None


@dataclass(frozen=True)
class GeneratorCheckpoint:
    '''A generator and everything needed to rebuild and describe it.'''

    model_name: str
    display_name: str
    architecture: Architecture
    conditional: bool
    seed: int
    generator: Generator
    history: tuple[EpochStats, ...]


def run_name(model_name: str, seed: int) -> str:
    '''Return the directory name for one training run.'''
    return f'{model_name}_seed{seed}'


def save_checkpoint(checkpoint: GeneratorCheckpoint, path: Path) -> None:
    '''Write the generator weights and metadata to one file.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': CHECKPOINT_FORMAT,
        'model_name': checkpoint.model_name,
        'display_name': checkpoint.display_name,
        'architecture': checkpoint.architecture,
        'conditional': checkpoint.conditional,
        'latent_dim': checkpoint.generator.latent_dim,
        'seed': checkpoint.seed,
        'state_dict': checkpoint.generator.state_dict(),
        'history': [asdict(stats) for stats in checkpoint.history],
        # torch.__version__ is a str subclass that weights_only loading rejects
        'versions': {'digitgan': __version__, 'torch': str(torch.__version__)},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device | None = None) -> GeneratorCheckpoint:
    '''Rebuild a generator from a checkpoint file.

    Args:
        path: File written by save_checkpoint.
        device: Device to place the generator on. Defaults to CPU.

    Returns:
        The checkpoint with its generator in evaluation mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a checkpoint in the expected format.
    '''
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    # weights_only refuses arbitrary pickled objects, so a tampered file cannot run code
    payload: dict[str, Any] = torch.load(path, map_location=device or 'cpu', weights_only=True)
    if payload.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{path} is not a digitgan checkpoint in format {CHECKPOINT_FORMAT}')

    config = GanConfig(
        display_name=payload['display_name'],
        architecture=payload['architecture'],
        conditional=payload['conditional'],
    )
    generator = build_generator(config.architecture, config.conditional, payload['latent_dim'])
    generator.load_state_dict(payload['state_dict'])
    generator.to(device or 'cpu').eval()
    return GeneratorCheckpoint(
        model_name=payload['model_name'],
        display_name=config.display_name,
        architecture=config.architecture,
        conditional=config.conditional,
        seed=payload['seed'],
        generator=generator,
        history=tuple(EpochStats(**stats) for stats in payload['history']),
    )
