'''Tests for the metrics and the reference image sources.'''

import numpy as np
import pytest
import torch

from digitgan.baseline import ClassMeanBaseline
from digitgan.checkpoints import GeneratorCheckpoint
from digitgan.classifier import DigitClassifier
from digitgan.config import AppConfig
from digitgan.data import DigitImages
from digitgan.evaluation import (
    Evaluator,
    FeatureStatistics,
    MetricSummary,
    SourceMetrics,
    baseline_source,
    frechet_distance,
    gan_source,
    normalised_entropy,
    real_source,
    summarise,
)


@pytest.fixture(scope='module')
def evaluator(
    session_config: AppConfig, trained_classifier: DigitClassifier, test_data: DigitImages
) -> Evaluator:
    return Evaluator(trained_classifier, test_data, session_config.evaluation)


def make_metrics(key: str, seed: int | None, fid: float, correct: float | None) -> SourceMetrics:
    return SourceMetrics(
        key=key,
        display_name=key.title(),
        kind='gan' if seed is not None else 'reference',
        conditional=correct is not None,
        seed=seed,
        pixel_mse=0.4,
        fid=fid,
        recognisable_rate=0.9,
        class_balance=1.0,
        class_shares=(0.1,) * 10,
        correct_digit_rate=correct,
        confusion=None,
    )


def test_frechet_distance_of_identical_statistics_is_zero() -> None:
    features = np.random.default_rng(0).normal(size=(500, 4))
    statistics = FeatureStatistics.from_features(features)

    assert frechet_distance(statistics, statistics) == pytest.approx(0, abs=1e-8)


def test_frechet_distance_matches_the_closed_form_for_diagonal_gaussians() -> None:
    first = FeatureStatistics(np.array([0.0, 1.0]), np.diag([1.0, 4.0]))
    second = FeatureStatistics(np.array([3.0, 1.0]), np.diag([4.0, 9.0]))

    # Squared mean difference 9, plus (sqrt(1) - sqrt(4))^2 + (sqrt(4) - sqrt(9))^2 = 2
    assert frechet_distance(first, second) == pytest.approx(11.0)


def test_normalised_entropy_runs_from_zero_to_one() -> None:
    assert normalised_entropy(np.full(10, 0.1)) == pytest.approx(1.0)
    assert normalised_entropy(np.eye(10)[3]) == pytest.approx(0.0)


def test_real_source_returns_images_of_the_requested_digits(train_data: DigitImages) -> None:
    requested = torch.tensor([3, 3, 8, 0])

    images = real_source(train_data, same_digit=True, seed=0).sample(requested)

    # The synthetic digit d lights rows 2 + 2d and 3 + 2d
    lit_rows = images[:, 0, :, 14].argmax(dim=1)
    assert ((lit_rows - 2) // 2).tolist() == requested.tolist()


def test_real_source_of_any_digit_is_reproducible(train_data: DigitImages) -> None:
    source = real_source(train_data, same_digit=False, seed=0)
    labels = torch.zeros(20, dtype=torch.long)

    assert not source.conditional
    assert torch.equal(source.sample(labels), source.sample(labels))


def test_real_source_needs_images_of_every_requested_digit() -> None:
    data = DigitImages(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))

    with pytest.raises(ValueError, match='digit 5'):
        real_source(data, same_digit=True, seed=0).sample(torch.tensor([5]))


def test_real_images_beat_an_untrained_generator(
    evaluator: Evaluator,
    train_data: DigitImages,
    conditional_checkpoint: GeneratorCheckpoint,
) -> None:
    generator = conditional_checkpoint.generator

    real = evaluator.score(real_source(train_data, same_digit=True, seed=0))
    untrained = evaluator.score(gan_source('cgan', 'Conditional GAN', generator, 1, 0))

    assert real.fid < untrained.fid
    assert real.correct_digit_rate is not None
    assert untrained.correct_digit_rate is not None
    assert real.correct_digit_rate > 0.9 > untrained.correct_digit_rate
    assert real.class_balance > 0.95
    assert real.confusion is not None
    assert sum(map(sum, real.confusion)) == evaluator.config.sample_size


def test_unconditional_sources_have_no_correct_digit_rate(
    evaluator: Evaluator, train_data: DigitImages
) -> None:
    metrics = evaluator.score(real_source(train_data, same_digit=False, seed=0))

    assert metrics.correct_digit_rate is None
    assert metrics.confusion is None
    assert sum(metrics.class_shares) == pytest.approx(1.0)


def test_class_mean_beats_real_images_on_pixel_mse(
    evaluator: Evaluator, train_data: DigitImages
) -> None:
    baseline = evaluator.pixel_mse(baseline_source(ClassMeanBaseline.fit(train_data)))
    real = evaluator.pixel_mse(real_source(train_data, same_digit=True, seed=0))

    # The mean minimises expected squared error, and an independent real image scores about
    # twice that, so pixel MSE ranks a blurred average above perfect handwriting
    assert baseline < real
    assert real == pytest.approx(2 * baseline, rel=0.25)


def test_fid_tracker_is_deterministic(
    evaluator: Evaluator, conditional_checkpoint: GeneratorCheckpoint
) -> None:
    track = evaluator.fid_tracker(50)

    assert track(conditional_checkpoint.generator) == track(conditional_checkpoint.generator)


def test_summaries_group_seeds_and_keep_source_order() -> None:
    metrics = [
        make_metrics('real', None, fid=1.0, correct=0.99),
        make_metrics('cgan', 1, fid=10.0, correct=0.9),
        make_metrics('dcgan', 1, fid=30.0, correct=None),
        make_metrics('cgan', 2, fid=14.0, correct=0.8),
    ]

    summaries = summarise(metrics)

    assert [summary.key for summary in summaries] == ['real', 'cgan', 'dcgan']
    real, cgan, dcgan = summaries
    assert real.seeds == ()
    assert real.fid == MetricSummary(1.0, 0.0)
    assert cgan.seeds == (1, 2)
    assert cgan.fid.mean == pytest.approx(12.0)
    assert cgan.fid.std == pytest.approx(np.sqrt(8))
    assert cgan.correct_digit_rate is not None
    assert cgan.correct_digit_rate.mean == pytest.approx(0.85)
    assert dcgan.correct_digit_rate is None
