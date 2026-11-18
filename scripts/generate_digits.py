'''Write a number in generated handwriting with a trained conditional GAN.'''

import argparse
import logging
import secrets
from pathlib import Path

from digitgan.checkpoints import load_checkpoint
from digitgan.config import EnvironmentSettings, configure_logging, load_config
from digitgan.generation import generate_number, render_grid

logger = logging.getLogger('generate_digits')

MAX_SEED = 2**31 - 1


def parse_arguments() -> argparse.Namespace:
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', default='0123456789', help='Digits to write, such as 2026.')
    parser.add_argument(
        '--variations', type=int, default=1, help='How many times to write the number.'
    )
    parser.add_argument('--seed', type=int, help='Seed for the noise. Random when omitted.')
    parser.add_argument('--scale', type=int, default=4, help='Pixel enlargement factor.')
    parser.add_argument('--config_path', type=Path, help='Configuration file to read.')
    parser.add_argument(
        '--checkpoint_path', type=Path, help='Generator to use. Defaults to the API setting.'
    )
    parser.add_argument('--output', type=Path, help='PNG file to write.')
    return parser.parse_args()


def main() -> None:
    '''Generate the requested number and save it as a PNG.'''
    arguments = parse_arguments()
    settings = EnvironmentSettings()
    configure_logging(settings.log_level)
    if not arguments.digits.isdecimal() or not arguments.digits.isascii():
        raise SystemExit('--digits must contain only the digits 0 to 9')
    if arguments.variations < 1 or arguments.scale < 1:
        raise SystemExit('--variations and --scale must be at least 1')

    config = load_config(arguments.config_path, settings)
    checkpoint_path = arguments.checkpoint_path or config.api.checkpoint_path
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        seed = arguments.seed if arguments.seed is not None else secrets.randbelow(MAX_SEED)
        images = generate_number(
            checkpoint.generator,
            [int(digit) for digit in arguments.digits],
            arguments.variations,
            seed,
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error('%s', error)
        raise SystemExit(1) from error

    output = arguments.output or Path('outputs') / f'{arguments.digits}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    render_grid(images, scale=arguments.scale).save(output)
    logger.info('Saved %s with seed %d', output.as_posix(), seed)


if __name__ == '__main__':
    main()
