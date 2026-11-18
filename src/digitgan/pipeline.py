'''Entry points that the scripts call: train every model, then evaluate every run.'''

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from digitgan import plotting
from digitgan.baseline import ClassMeanBaseline
from digitgan.checkpoints import (
    GENERATOR_FILE,
    PROGRESS_FILE,
    EpochStats,
    GeneratorCheckpoint,
    load_checkpoint,
    run_name,
)
from digitgan.classifier import (
    CLASSIFIER_FILE,
    DigitClassifier,
    accuracy,
    load_classifier,
    save_classifier,
    train_classifier,
)
from digitgan.config import AppConfig
from digitgan.data import NUM_CLASSES, DigitImages, balanced_labels
from digitgan.evaluation import (
    Evaluator,
    ImageSource,
    SourceMetrics,
    SourceSummary,
    baseline_source,
    gan_source,
    real_source,
    summarise,
)
from digitgan.generation import generate_images, grid_inputs, interpolate_styles, to_uint8
from digitgan.reporting import (
    FIGURE_DIR_NAME,
    METRICS_FILE,
    REPORT_FILE,
    ReportData,
    write_evaluation_report,
    write_metrics_json,
)
from digitgan.training import save_run, train_gan

logger = logging.getLogger(__name__)

CLASSIFIER_DIR = 'classifier'
INTERPOLATION_STEPS = 10


def classifier_path(config: AppConfig) -> Path:
    '''Return where the evaluation classifier is stored.'''
    return config.outputs.run_dir / CLASSIFIER_DIR / CLASSIFIER_FILE


def prepare_classifier(
    config: AppConfig,
    train_data: DigitImages,
    test_data: DigitImages,
    device: torch.device,
    retrain: bool = False,
) -> DigitClassifier:
    '''Load the evaluation classifier, training it first if it does not exist yet.

    Reusing one classifier keeps every FID and accuracy figure comparable across runs.
    '''
    path = classifier_path(config)
    if path.is_file() and not retrain:
        classifier, test_accuracy = load_classifier(path, device)
        logger.info('Loaded classifier with test accuracy %.2f%%', 100 * test_accuracy)
        return classifier

    logger.info('Training classifier')
    classifier = train_classifier(train_data, config.classifier, device)
    test_accuracy = accuracy(classifier, test_data)
    save_classifier(classifier, test_accuracy, path)
    logger.info('Classifier test accuracy %.2f%%', 100 * test_accuracy)
    return classifier


def run_training(
    config: AppConfig,
    train_data: DigitImages,
    test_data: DigitImages,
    model_names: Sequence[str],
    seeds: Sequence[int],
    retrain_classifier: bool = False,
) -> list[Path]:
    '''Train each model once per seed and save every run.

    Runs are ordered by seed, so that an interrupted session leaves complete seeds behind.

    Returns:
        The directories of the saved runs.
    '''
    unknown = sorted(set(model_names) - set(config.models))
    if unknown:
        raise ValueError(f'Unknown models: {", ".join(unknown)}')

    device = config.resolve_device()
    classifier = prepare_classifier(config, train_data, test_data, device, retrain_classifier)
    tracker = Evaluator(classifier, test_data, config.evaluation).fid_tracker(
        config.training.tracking_sample_size
    )

    run_dirs = []
    for seed in seeds:
        for name in model_names:
            logger.info('Training %s with seed %d', name, seed)
            run = train_gan(
                name, config.models[name], config.training, train_data, seed, device, tracker
            )
            run_dirs.append(save_run(run, config.outputs.run_dir))
            logger.info('Saved %s', run_dirs[-1].as_posix())
    return run_dirs


@dataclass(frozen=True)
class EvaluationOutcome:
    '''Where the evaluation wrote its outputs, and the headline numbers.'''

    report_path: Path
    metrics_path: Path
    summaries: list[SourceSummary]


def load_runs(config: AppConfig, device: torch.device) -> list[GeneratorCheckpoint]:
    '''Load every saved run for the configured models and seeds, ordered by model.'''
    checkpoints = []
    for name in config.models:
        for seed in config.training.seeds:
            path = config.outputs.run_dir / run_name(name, seed) / GENERATOR_FILE
            if path.is_file():
                checkpoints.append(load_checkpoint(path, device))
            else:
                logger.warning('No run for %s with seed %d', name, seed)
    return checkpoints


def run_evaluation(
    config: AppConfig, train_data: DigitImages, test_data: DigitImages
) -> EvaluationOutcome:
    '''Score every saved run against the references, then draw figures and write the report.

    Raises:
        FileNotFoundError: If the classifier or every training run is missing.
    '''
    device = config.resolve_device()
    classifier, classifier_accuracy = load_classifier(classifier_path(config), device)
    logger.info('Loaded classifier with test accuracy %.2f%%', 100 * classifier_accuracy)
    checkpoints = load_runs(config, device)
    if not checkpoints:
        raise FileNotFoundError(
            f'No training runs in {config.outputs.run_dir}. Run scripts/train_models.py first.'
        )

    evaluator = Evaluator(classifier, test_data, config.evaluation)
    seed = config.evaluation.seed
    references = [
        real_source(train_data, same_digit=True, seed=seed),
        real_source(train_data, same_digit=False, seed=seed),
        baseline_source(ClassMeanBaseline.fit(train_data)),
    ]
    gans = [
        gan_source(c.model_name, c.display_name, c.generator, c.seed, seed) for c in checkpoints
    ]

    metrics = []
    for source in [*references, *gans]:
        label = source.display_name if source.seed is None else f'{source.key} seed {source.seed}'
        logger.info('Scoring %s', label)
        metrics.append(evaluator.score(source))
    summaries = summarise(metrics)

    logger.info('Drawing figures')
    report_dir = config.outputs.report_dir
    figures = draw_figures(config, references, checkpoints, metrics, summaries, evaluator)

    data = ReportData(
        config=config,
        train_size=len(train_data),
        test_size=len(test_data),
        classifier_accuracy=classifier_accuracy,
        metrics=metrics,
        summaries=summaries,
        checkpoints=checkpoints,
        figures=figures,
    )
    return EvaluationOutcome(
        report_path=write_evaluation_report(data, report_dir / REPORT_FILE),
        metrics_path=write_metrics_json(data, report_dir / METRICS_FILE),
        summaries=summaries,
    )


def draw_figures(
    config: AppConfig,
    references: Sequence[ImageSource],
    checkpoints: Sequence[GeneratorCheckpoint],
    metrics: Sequence[SourceMetrics],
    summaries: Sequence[SourceSummary],
    evaluator: Evaluator,
) -> list[tuple[str, Path]]:
    '''Draw every figure into the report directory.

    The first seed of each model is its showcase run for sample pictures. Charts use every seed.

    Args:
        config: Project configuration.
        references: Real images of the same digit, real images of any digit, and the baseline.
        checkpoints: Every loaded run, ordered by model.
        metrics: Per-run metrics from the evaluator.
        summaries: Metrics aggregated over seeds.
        evaluator: Evaluator used for the real-image FID reference line.

    Returns:
        Figure titles and paths relative to the report directory, in display order.
    '''
    report_dir = config.outputs.report_dir
    figure_dir = report_dir / FIGURE_DIR_NAME
    seed = config.evaluation.seed
    colours = plotting.model_colours(list(config.models))
    names = {key: model.display_name for key, model in config.models.items()}
    showcase: dict[str, GeneratorCheckpoint] = {}
    histories: dict[str, list[list[EpochStats]]] = {}
    for checkpoint in checkpoints:
        showcase.setdefault(checkpoint.model_name, checkpoint)
        histories.setdefault(checkpoint.model_name, []).append(list(checkpoint.history))

    figures: list[tuple[str, Path]] = []

    def add(title: str, path: Path) -> None:
        figures.append((title, path.relative_to(report_dir)))

    digits = torch.arange(NUM_CLASSES)
    real_same_digit, real_any_digit, baseline = references
    rows = [
        ('Real images', 'Training images of 0 to 9', to_uint8(real_same_digit.sample(digits))),
        ('Class-mean baseline', 'Average of each digit', to_uint8(baseline.sample(digits))),
    ]
    for checkpoint in showcase.values():
        source = gan_source(
            checkpoint.model_name,
            checkpoint.display_name,
            checkpoint.generator,
            checkpoint.seed,
            seed,
        )
        caption = 'Asked for 0 to 9' if checkpoint.conditional else 'Random digits, no control'
        rows.append((checkpoint.display_name, caption, to_uint8(source.sample(digits))))
    add(
        'Samples from each source',
        plotting.plot_model_comparison(rows, figure_dir / 'model_comparison.png'),
    )

    for key, checkpoint in showcase.items():
        noise, labels = grid_inputs(checkpoint.generator, seed)
        grid = to_uint8(generate_images(checkpoint.generator, noise, labels))
        title = f'{checkpoint.display_name} samples'
        if checkpoint.conditional:
            title += ': one noise vector per row, digits 0 to 9 across'
        picture = grid.reshape(NUM_CLASSES, NUM_CLASSES, *grid.shape[1:])
        add(title, plotting.save_picture(picture, figure_dir / f'{key}_samples.png'))

    for key, checkpoint in showcase.items():
        if not checkpoint.conditional:
            continue
        progress_path = config.outputs.run_dir / run_name(key, checkpoint.seed) / PROGRESS_FILE
        if progress_path.is_file():
            with np.load(progress_path) as progress:
                frames = progress['frames']
            animation = plotting.save_training_animation(
                frames, figure_dir / f'{key}_training.gif', animation_epochs(len(frames) - 1)
            )
            add(f'{checkpoint.display_name} during training', animation)
        steps = to_uint8(interpolate_styles(checkpoint.generator, seed, INTERPOLATION_STEPS))
        add(
            f'{checkpoint.display_name}: moving between two noise vectors for each digit',
            plotting.save_picture(steps, figure_dir / f'{key}_interpolation.png'),
        )

    add(
        'Pixel MSE, compared like for like',
        plotting.plot_pixel_mse(summaries, colours, figure_dir / 'pixel_mse.png'),
    )
    if any(stats.fid is not None for runs in histories.values() for stats in runs[0]):
        real_images = real_any_digit.sample(balanced_labels(config.training.tracking_sample_size))
        chart = plotting.plot_fid_during_training(
            histories,
            names,
            colours,
            evaluator.fid(real_images),
            figure_dir / 'fid_during_training.png',
        )
        add('FID during training', chart)
    add(
        'Training losses',
        plotting.plot_training_losses(histories, names, figure_dir / 'training_losses.png'),
    )

    unconditional = [s for s in summaries if not s.conditional]
    if unconditional:
        shares = [(s.key, s.display_name, _mean_shares(metrics, s.key)) for s in unconditional]
        chart = plotting.plot_class_distribution(
            shares, colours, figure_dir / 'class_distribution.png'
        )
        add('Digits produced without conditioning', chart)

    for key, checkpoint in showcase.items():
        if not checkpoint.conditional:
            continue
        counts = [np.array(m.confusion) for m in metrics if m.key == key and m.confusion]
        chart = plotting.plot_confusion(
            np.sum(counts, axis=0),
            f'{checkpoint.display_name}, all seeds (row %)',
            figure_dir / f'{key}_confusion.png',
        )
        add(f'{checkpoint.display_name}: requested digit against classified digit', chart)
    return figures


def animation_epochs(total: int) -> list[int]:
    '''Choose animation frames: every early epoch, where change is fastest, then fewer.'''
    early = range(min(total, 5) + 1)
    middle = range(6, min(total, 12) + 1, 2)
    late = range(15, total + 1, 5)
    return sorted({*early, *middle, *late, total})


def _mean_shares(metrics: Sequence[SourceMetrics], key: str) -> list[float]:
    rows = [item.class_shares for item in metrics if item.key == key]
    return [float(value) for value in np.mean(rows, axis=0)]
