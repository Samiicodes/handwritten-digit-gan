'''Score every trained model, draw the figures, and write the evaluation report.'''

import argparse
import logging
import shutil
from pathlib import Path

from digitgan.config import AppConfig, EnvironmentSettings, configure_logging, load_config
from digitgan.data import load_mnist
from digitgan.evaluation import SourceSummary
from digitgan.pipeline import EvaluationOutcome, run_evaluation
from digitgan.reporting import FIGURE_DIR_NAME, format_fid, format_mse, format_rate

logger = logging.getLogger('evaluate_models')

DOCS_DIR = Path('docs')


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument(
        '--sample_size', type=int, help='Override the number of generated images per run.'
    )
    parser.add_argument(
        '--publish_docs',
        action='store_true',
        help='Copy the evaluation report and its figures into docs/.',
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, sample_size: int | None) -> AppConfig:
    '''Return the configuration with command-line overrides applied.'''
    if sample_size is None:
        return config
    evaluation = config.evaluation.model_copy(update={'sample_size': sample_size})
    return config.model_copy(update={'evaluation': evaluation})


def describe(summary: SourceSummary) -> str:
    '''One line of headline numbers for a source.'''
    line = (
        f'{summary.display_name}: pixel MSE {format_mse(summary.pixel_mse.mean)}, '
        f'FID {format_fid(summary.fid.mean)}'
    )
    if summary.correct_digit_rate is not None:
        line += f', correct digit {format_rate(summary.correct_digit_rate.mean)}'
    return line


def publish_docs(outcome: EvaluationOutcome) -> None:
    '''Copy the generated report and figures into the versioned docs directory.'''
    report_dir = outcome.report_path.parent
    shutil.copy2(outcome.report_path, DOCS_DIR / outcome.report_path.name)
    shutil.copytree(report_dir / FIGURE_DIR_NAME, DOCS_DIR / FIGURE_DIR_NAME, dirs_exist_ok=True)
    logger.info('Published report to %s', DOCS_DIR.as_posix())


def main() -> None:
    '''Evaluate the saved runs and print the headline numbers.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level)
    config = apply_overrides(load_config(arguments.config_path, settings), arguments.sample_size)

    logger.info('Loading dataset')
    train_data = load_mnist(config.data.root, 'train')
    test_data = load_mnist(config.data.root, 'test')

    try:
        outcome = run_evaluation(config, train_data, test_data)
    except FileNotFoundError as error:
        logger.error('%s', error)
        raise SystemExit(1) from error

    for summary in outcome.summaries:
        logger.info('%s', describe(summary))
    logger.info('Report written to %s', outcome.report_path.as_posix())
    if arguments.publish_docs:
        publish_docs(outcome)
    logger.info('Done')


if __name__ == '__main__':
    main()
