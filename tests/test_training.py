'''Tests for the adversarial training loop and for checkpoints.'''

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from digitgan.checkpoints import GENERATOR_FILE, PROGRESS_FILE, load_checkpoint
from digitgan.config import AppConfig, GanConfig
from digitgan.data import DigitImages
from digitgan.generation import generate_images, sample_noise
from digitgan.models import Generator
from digitgan.training import TrainingRun, save_run, train_gan


def train(
    app_config: AppConfig, data: DigitImages, name: str = 'cgan', seed: int = 1
) -> TrainingRun:
    device = torch.device('cpu')
    return train_gan(name, app_config.models[name], app_config.training, data, seed, device)


@pytest.mark.parametrize('name', ['vanilla_gan', 'dcgan', 'cgan'])
def test_training_records_every_epoch(
    app_config: AppConfig, train_data: DigitImages, name: str
) -> None:
    training = app_config.training.model_copy(update={'epochs': 2})

    run = train_gan(
        name, app_config.models[name], training, train_data.head(200), 1, torch.device('cpu')
    )

    history = run.checkpoint.history
    assert [stats.epoch for stats in history] == [1, 2]
    assert all(math.isfinite(stats.generator_loss) for stats in history)
    assert all(0 < stats.real_score < 1 and 0 < stats.fake_score < 1 for stats in history)
    # One progress frame before training and one after each epoch
    assert run.progress.shape == (3, 100, 28, 28)
    assert run.progress.dtype == np.uint8
    assert not run.checkpoint.generator.training


def test_tracker_runs_on_tracking_epochs_and_the_last_epoch(
    app_config: AppConfig, train_data: DigitImages
) -> None:
    training = app_config.training.model_copy(update={'epochs': 3, 'tracking_interval': 2})
    calls: list[bool] = []

    def tracker(generator: Generator) -> float:
        calls.append(generator.training)
        return 12.5

    run = train_gan(
        'dcgan',
        app_config.models['dcgan'],
        training,
        train_data.head(100),
        1,
        torch.device('cpu'),
        tracker,
    )

    assert [stats.fid for stats in run.checkpoint.history] == [None, 12.5, 12.5]
    assert len(calls) == 2


def test_the_same_seed_gives_the_same_generator(
    app_config: AppConfig, train_data: DigitImages
) -> None:
    data = train_data.head(100)

    first = train(app_config, data, seed=3).checkpoint.generator.state_dict()
    second = train(app_config, data, seed=3).checkpoint.generator.state_dict()
    other = train(app_config, data, seed=4).checkpoint.generator.state_dict()

    assert all(torch.equal(first[key], second[key]) for key in first)
    assert not all(torch.equal(first[key], other[key]) for key in first)


def test_saved_run_reloads_with_identical_outputs(
    app_config: AppConfig, train_data: DigitImages, tmp_path: Path
) -> None:
    run = train(app_config, train_data.head(100))

    directory = save_run(run, tmp_path)
    loaded = load_checkpoint(directory / GENERATOR_FILE)

    assert directory.name == 'cgan_seed1'
    assert loaded.model_name == 'cgan'
    assert loaded.conditional
    assert loaded.history == run.checkpoint.history
    noise, labels = sample_noise(4, 100, seed=0), torch.arange(4)
    assert torch.equal(
        generate_images(loaded.generator, noise, labels),
        generate_images(run.checkpoint.generator, noise, labels),
    )
    with np.load(directory / PROGRESS_FILE) as progress:
        assert np.array_equal(progress['frames'], run.progress)


def test_loading_a_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / 'missing.pt')


def test_loading_a_foreign_file_raises(tmp_path: Path) -> None:
    path = tmp_path / 'other.pt'
    torch.save({'weights': torch.zeros(2)}, path)

    with pytest.raises(ValueError, match='not a digitgan checkpoint'):
        load_checkpoint(path)


def test_model_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        GanConfig.model_validate(
            {'display_name': 'x', 'architecture': 'convolutional', 'conditional': 'maybe'}
        )
