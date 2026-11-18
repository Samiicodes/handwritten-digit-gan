'''Shared fixtures: a small configuration and synthetic digits that need no download.'''

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from digitgan.checkpoints import GeneratorCheckpoint
from digitgan.classifier import DigitClassifier, train_classifier
from digitgan.config import AppConfig
from digitgan.data import IMAGE_SIZE, NUM_CLASSES, DigitImages
from digitgan.models import build_generator

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'


def make_digits(count: int, seed: int) -> DigitImages:
    '''Synthetic digits: each class lights a different horizontal stripe, plus a little noise.'''
    rng = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % NUM_CLASSES
    images = torch.full((count, 1, IMAGE_SIZE, IMAGE_SIZE), -1.0)
    rows = 2 + 2 * labels
    index = torch.arange(count)
    images[index, 0, rows, 4:24] = 1.0
    images[index, 0, rows + 1, 4:24] = 1.0
    images += 0.1 * torch.randn(images.shape, generator=rng)
    return DigitImages(images.clamp(-1, 1), labels)


def build_config(base_dir: Path) -> AppConfig:
    '''The repository configuration, shrunk so that every stage runs in seconds.'''
    raw: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    raw['device'] = 'cpu'
    raw['data']['root'] = str(base_dir / 'data')
    raw['training'].update(
        seeds=[1, 2], epochs=1, batch_size=50, tracking_interval=1, tracking_sample_size=50
    )
    raw['classifier'].update(epochs=2, batch_size=50)
    raw['evaluation'].update(sample_size=100, pixel_mse_size=50)
    raw['outputs'] = {'run_dir': str(base_dir / 'runs'), 'report_dir': str(base_dir / 'reports')}
    raw['api']['checkpoint_path'] = str(base_dir / 'runs' / 'cgan_seed1' / 'generator.pt')
    return AppConfig.model_validate(raw)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return build_config(tmp_path)


@pytest.fixture(scope='session')
def session_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    '''A configuration shared by fixtures that are expensive enough to build once.'''
    return build_config(tmp_path_factory.mktemp('session'))


@pytest.fixture(scope='session')
def train_data() -> DigitImages:
    return make_digits(1000, seed=0)


@pytest.fixture(scope='session')
def test_data() -> DigitImages:
    return make_digits(200, seed=1)


@pytest.fixture(scope='session')
def trained_classifier(session_config: AppConfig, train_data: DigitImages) -> DigitClassifier:
    return train_classifier(train_data, session_config.classifier, torch.device('cpu'))


@pytest.fixture
def conditional_checkpoint() -> GeneratorCheckpoint:
    '''An untrained conditional generator, enough to exercise everything but image quality.'''
    torch.manual_seed(0)
    return GeneratorCheckpoint(
        model_name='cgan',
        display_name='Conditional GAN',
        architecture='convolutional',
        conditional=True,
        seed=1,
        generator=build_generator('convolutional', conditional=True, latent_dim=100).eval(),
        history=(),
    )
