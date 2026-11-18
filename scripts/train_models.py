'''Train the evaluation classifier and every GAN in the comparison.'''

import argparse
import logging
from pathlib import Path

from digitgan.config import AppConfig, EnvironmentSettings, configure_logging, load_config
from digitgan.data import load_mnist
from digitgan.pipeline import run_training

logger = logging.getLogger('train_models')


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument('--models', nargs='+', help='Models to train. Defaults to all of them.')
    parser.add_argument('--seeds', nargs='+', type=int, help='Seeds to train. Defaults to all.')
    parser.add_argument('--epochs', type=int, help='Override the number of epochs.')
    parser.add_argument(
        '--train_size',
        type=int,
        help='Train on the first N training images only, for quick checks.',
    )
    parser.add_argument(
        '--retrain_classifier',
        action='store_true',
        help='Train the evaluation classifier again even if it exists.',
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, epochs: int | None) -> AppConfig:
    '''Return the configuration with command-line overrides applied.'''
    if epochs is None:
        return config
    training = config.training.model_copy(update={'epochs': epochs})
    return config.model_copy(update={'training': training})


def main() -> None:
    '''Load MNIST and train the requested models.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level)
    config = apply_overrides(load_config(arguments.config_path, settings), arguments.epochs)

    logger.info('Loading dataset')
    train_data = load_mnist(config.data.root, 'train')
    test_data = load_mnist(config.data.root, 'test')
    if arguments.train_size:
        train_data = train_data.head(arguments.train_size)
    logger.info(
        'Loaded %s training and %s test images', f'{len(train_data):,}', f'{len(test_data):,}'
    )

    try:
        run_training(
            config,
            train_data,
            test_data,
            model_names=arguments.models or list(config.models),
            seeds=arguments.seeds or config.training.seeds,
            retrain_classifier=arguments.retrain_classifier,
        )
    except ValueError as error:
        logger.error('%s', error)
        raise SystemExit(1) from error
    logger.info('Done')


if __name__ == '__main__':
    main()
