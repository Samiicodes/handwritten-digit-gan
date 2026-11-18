'''Metrics for generated digits, and the reference points that make them interpretable.

Every image source, whether a GAN, the baseline, or real data, is scored by the same
Evaluator with the same frozen classifier and the same test images.
'''

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import torch
from torch import Tensor

from digitgan.baseline import ClassMeanBaseline
from digitgan.classifier import DigitClassifier, classify
from digitgan.config import EvaluationConfig
from digitgan.data import NUM_CLASSES, DigitImages, balanced_labels
from digitgan.generation import generate_images, sample_noise
from digitgan.models import Generator

SourceKind = Literal['reference', 'baseline', 'gan']
# Maps requested digits to images in [-1, 1]. Unconditional sources use only the count.
Sampler = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ImageSource:
    '''Anything that produces digit images: a GAN, the baseline, or real data.'''

    key: str
    display_name: str
    kind: SourceKind
    # Whether the output depends on the requested digit
    conditional: bool
    sample: Sampler
    seed: int | None = None


def gan_source(
    key: str, display_name: str, generator: Generator, training_seed: int, sampling_seed: int
) -> ImageSource:
    '''Wrap a trained generator. The same sampling seed gives the same noise on every call.'''

    def sample(labels: Tensor) -> Tensor:
        noise = sample_noise(len(labels), generator.latent_dim, sampling_seed)
        return generate_images(generator, noise, labels if generator.conditional else None)

    return ImageSource(key, display_name, 'gan', generator.conditional, sample, training_seed)


def baseline_source(baseline: ClassMeanBaseline) -> ImageSource:
    '''Wrap the class-mean baseline.'''
    return ImageSource('baseline', 'Class-mean baseline', 'baseline', True, baseline.generate)


def real_source(train_data: DigitImages, same_digit: bool, seed: int) -> ImageSource:
    '''Draw real training images, either of the requested digit or of any digit.

    Real images are what a perfect generator would produce, so these sources show the best
    score each metric can give, and whether a metric can tell real images from generated ones.
    '''

    def sample(labels: Tensor) -> Tensor:
        rng = torch.Generator().manual_seed(seed)
        if not same_digit:
            return train_data.images[torch.randint(len(train_data), (len(labels),), generator=rng)]
        indices = torch.empty(len(labels), dtype=torch.long)
        for digit in range(NUM_CLASSES):
            wanted = torch.nonzero(labels == digit).squeeze(1)
            if not len(wanted):
                continue
            pool = torch.nonzero(train_data.labels == digit).squeeze(1)
            if not len(pool):
                raise ValueError(f'no training images of digit {digit}')
            indices[wanted] = pool[torch.randint(len(pool), (len(wanted),), generator=rng)]
        return train_data.images[indices]

    if same_digit:
        return ImageSource('real_same_digit', 'Real images, same digit', 'reference', True, sample)
    return ImageSource('real_any_digit', 'Real images, any digit', 'reference', False, sample)


@dataclass(frozen=True)
class FeatureStatistics:
    '''Mean and covariance of classifier features, the two moments that FID compares.'''

    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> Self:
        '''Estimate the statistics from an array of shape (n, features).'''
        return cls(features.mean(axis=0), np.cov(features, rowvar=False))


def frechet_distance(first: FeatureStatistics, second: FeatureStatistics) -> float:
    '''Frechet distance between two Gaussians fitted to feature sets, as used by FID.

    The trace of sqrt(C1 C2) is computed from the eigenvalues of C1 C2. For covariance
    matrices they are real and non-negative, so rounding errors are clipped away.
    '''
    difference = first.mean - second.mean
    eigenvalues = np.linalg.eigvals(first.covariance @ second.covariance)
    trace_sqrt = np.sqrt(np.clip(eigenvalues.real, 0, None)).sum()
    distance = (
        difference @ difference
        + np.trace(first.covariance)
        + np.trace(second.covariance)
        - 2 * trace_sqrt
    )
    return float(max(distance, 0.0))


def normalised_entropy(shares: np.ndarray) -> float:
    '''Entropy of a distribution over the ten digits, scaled so that uniform gives 1.0.'''
    nonzero = shares[shares > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(len(shares)))


@dataclass(frozen=True)
class SourceMetrics:
    '''Every metric for one image source, and for GANs one training seed.'''

    key: str
    display_name: str
    kind: SourceKind
    conditional: bool
    seed: int | None
    # Coursework metric: squared pixel difference from test images, lower is closer
    pixel_mse: float
    # Frechet distance to the test set in classifier feature space, lower is better
    fid: float
    # Share of images the classifier assigns to one digit with high confidence
    recognisable_rate: float
    # Normalised entropy of the predicted digits; 1.0 means all ten appear equally often
    class_balance: float
    class_shares: tuple[float, ...]
    # Conditional sources only: share classified as the digit that was asked for
    correct_digit_rate: float | None
    # Conditional sources only: counts of requested digit (rows) against predicted digit
    confusion: tuple[tuple[int, ...], ...] | None


class Evaluator:
    '''Scores image sources against the test split with a frozen classifier.'''

    def __init__(
        self, classifier: DigitClassifier, test_data: DigitImages, config: EvaluationConfig
    ) -> None:
        self.classifier = classifier
        self.test_data = test_data
        self.config = config
        self.reference = FeatureStatistics.from_features(
            classify(classifier, test_data.images).features
        )

    def fid(self, images: Tensor) -> float:
        '''FID between images and the test split, in the classifier's feature space.'''
        features = classify(self.classifier, images).features
        return frechet_distance(FeatureStatistics.from_features(features), self.reference)

    def pixel_mse(self, source: ImageSource) -> float:
        '''Score a source with the coursework protocol.

        Each of the first test images is compared with one generated image. Conditional sources
        are asked for the digit of the test image they are compared with. Unconditional sources
        cannot be, so their images are compared with test images of arbitrary digits.
        '''
        test = self.test_data.head(self.config.pixel_mse_size)
        generated = source.sample(test.labels)
        return float(((generated - test.images) ** 2).mean())

    def score(self, source: ImageSource) -> SourceMetrics:
        '''Compute every metric for one source.'''
        requested = balanced_labels(self.config.sample_size)
        images = source.sample(requested)
        outputs = classify(self.classifier, images)
        predicted = outputs.probabilities.argmax(axis=1)
        shares = np.bincount(predicted, minlength=NUM_CLASSES) / len(predicted)

        correct_digit_rate, confusion = None, None
        if source.conditional:
            correct_digit_rate = float((predicted == requested.numpy()).mean())
            counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
            np.add.at(counts, (requested.numpy(), predicted), 1)
            confusion = tuple(tuple(int(count) for count in row) for row in counts)

        return SourceMetrics(
            key=source.key,
            display_name=source.display_name,
            kind=source.kind,
            conditional=source.conditional,
            seed=source.seed,
            pixel_mse=self.pixel_mse(source),
            fid=frechet_distance(FeatureStatistics.from_features(outputs.features), self.reference),
            recognisable_rate=float(
                (outputs.probabilities.max(axis=1) >= self.config.confidence_threshold).mean()
            ),
            class_balance=normalised_entropy(shares),
            class_shares=tuple(float(share) for share in shares),
            correct_digit_rate=correct_digit_rate,
            confusion=confusion,
        )

    def fid_tracker(self, sample_size: int) -> Callable[[Generator], float]:
        '''Return a cheap FID measurement on a fixed sample, for tracking during training.'''
        labels = balanced_labels(sample_size)

        def track(generator: Generator) -> float:
            noise = sample_noise(sample_size, generator.latent_dim, self.config.seed)
            images = generate_images(generator, noise, labels if generator.conditional else None)
            return self.fid(images)

        return track


@dataclass(frozen=True)
class MetricSummary:
    '''Mean and sample standard deviation of one metric across training seeds.'''

    mean: float
    std: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Self:
        '''Summarise values. The spread of a single value is zero.'''
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        return cls(statistics.fmean(values), spread)


@dataclass(frozen=True)
class SourceSummary:
    '''Metrics for one source, aggregated over its training seeds.'''

    key: str
    display_name: str
    kind: SourceKind
    conditional: bool
    seeds: tuple[int, ...]
    pixel_mse: MetricSummary
    fid: MetricSummary
    recognisable_rate: MetricSummary
    class_balance: MetricSummary
    correct_digit_rate: MetricSummary | None


def summarise(metrics: Sequence[SourceMetrics]) -> list[SourceSummary]:
    '''Group per-seed metrics by source, keeping the order in which sources first appear.'''
    groups: dict[str, list[SourceMetrics]] = {}
    for item in metrics:
        groups.setdefault(item.key, []).append(item)

    summaries = []
    for items in groups.values():
        first = items[0]
        correct = [item.correct_digit_rate for item in items if item.correct_digit_rate is not None]
        summaries.append(
            SourceSummary(
                key=first.key,
                display_name=first.display_name,
                kind=first.kind,
                conditional=first.conditional,
                seeds=tuple(item.seed for item in items if item.seed is not None),
                pixel_mse=MetricSummary.from_values([item.pixel_mse for item in items]),
                fid=MetricSummary.from_values([item.fid for item in items]),
                recognisable_rate=MetricSummary.from_values(
                    [item.recognisable_rate for item in items]
                ),
                class_balance=MetricSummary.from_values([item.class_balance for item in items]),
                correct_digit_rate=MetricSummary.from_values(correct) if correct else None,
            )
        )
    return summaries
