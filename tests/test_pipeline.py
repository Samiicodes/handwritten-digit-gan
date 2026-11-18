'''End-to-end tests: train every model on synthetic digits, then evaluate and report.'''

import json
import logging
import re

import pytest
import torch

from digitgan.classifier import DigitClassifier, save_classifier
from digitgan.config import AppConfig
from digitgan.data import DigitImages
from digitgan.pipeline import (
    EvaluationOutcome,
    animation_epochs,
    classifier_path,
    prepare_classifier,
    run_evaluation,
    run_training,
)

Evaluated = tuple[AppConfig, EvaluationOutcome]


@pytest.fixture(scope='module')
def evaluated(
    session_config: AppConfig, train_data: DigitImages, test_data: DigitImages
) -> Evaluated:
    config = session_config
    run_training(config, train_data, test_data, list(config.models), config.training.seeds)
    return config, run_evaluation(config, train_data, test_data)


def test_training_saves_one_run_per_model_and_seed(evaluated: Evaluated) -> None:
    config, _ = evaluated

    names = sorted(path.name for path in config.outputs.run_dir.iterdir())

    assert names == [
        'cgan_seed1',
        'cgan_seed2',
        'classifier',
        'dcgan_seed1',
        'dcgan_seed2',
        'vanilla_gan_seed1',
        'vanilla_gan_seed2',
    ]


def test_report_covers_every_source_and_links_existing_figures(evaluated: Evaluated) -> None:
    _, outcome = evaluated

    report = outcome.report_path.read_text(encoding='utf-8')
    links = re.findall(r'\]\((figures/[^)]+)\)', report)

    for name in ('Real images, same digit', 'Class-mean baseline', 'Vanilla GAN', 'DCGAN'):
        assert f'| {name} |' in report
    assert '| Conditional GAN | 2 |' in report
    assert len(links) == 11
    assert all((outcome.report_path.parent / link).is_file() for link in links)


def test_metrics_json_holds_every_run(evaluated: Evaluated) -> None:
    _, outcome = evaluated

    payload = json.loads(outcome.metrics_path.read_text(encoding='utf-8'))

    # Three references plus three GANs with two seeds each
    assert len(payload['runs']) == 9
    assert [summary['key'] for summary in payload['summaries']] == [
        'real_same_digit',
        'real_any_digit',
        'baseline',
        'vanilla_gan',
        'dcgan',
        'cgan',
    ]


def test_existing_classifier_is_reused(
    evaluated: Evaluated,
    train_data: DigitImages,
    test_data: DigitImages,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config, _ = evaluated

    with caplog.at_level(logging.INFO, logger='digitgan.pipeline'):
        prepare_classifier(config, train_data, test_data, torch.device('cpu'))

    assert 'Loaded classifier' in caplog.text


def test_unknown_models_are_rejected(
    app_config: AppConfig, train_data: DigitImages, test_data: DigitImages
) -> None:
    with pytest.raises(ValueError, match='Unknown models: transformer'):
        run_training(app_config, train_data, test_data, ['transformer'], [1])


def test_evaluation_without_runs_explains_what_to_run(
    app_config: AppConfig,
    train_data: DigitImages,
    test_data: DigitImages,
    trained_classifier: DigitClassifier,
) -> None:
    save_classifier(trained_classifier, 0.9, classifier_path(app_config))

    with pytest.raises(FileNotFoundError, match=r'train_models\.py'):
        run_evaluation(app_config, train_data, test_data)


def test_animation_shows_every_early_epoch_then_fewer() -> None:
    assert animation_epochs(50) == [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    assert animation_epochs(1) == [0, 1]
