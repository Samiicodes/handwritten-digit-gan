'''Typed configuration loaded from config/config.yaml and environment variables.'''

import logging
from pathlib import Path
from typing import Any, Literal, Self

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Architecture = Literal['fully_connected', 'convolutional']
DeviceName = Literal['auto', 'cpu', 'cuda']

CLI_LOG_FORMAT = '%(message)s'
SERVICE_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'

# Third-party loggers that print lines unrelated to the pipeline stages
_NOISY_LOGGERS = ('matplotlib', 'PIL', 'httpx')


class StrictModel(BaseModel):
    '''Base class that rejects unknown keys so typos in the YAML file fail fast.'''

    model_config = ConfigDict(extra='forbid', frozen=True)


class DataConfig(StrictModel):
    '''Where MNIST is stored.'''

    root: Path


class TrainingConfig(StrictModel):
    '''Settings shared by every GAN, so that results differ only by model.'''

    seeds: tuple[int, ...] = Field(min_length=1)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    adam_betas: tuple[float, float]
    latent_dim: int = Field(ge=1)
    real_label_target: float = Field(gt=0.5, le=1)
    tracking_interval: int = Field(ge=1)
    tracking_sample_size: int = Field(ge=10)

    @model_validator(mode='after')
    def _seeds_are_unique(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be unique')
        return self


class GanConfig(StrictModel):
    '''What distinguishes one GAN from another in the comparison.'''

    display_name: str = Field(min_length=1)
    architecture: Architecture
    conditional: bool


class ClassifierConfig(StrictModel):
    '''Training settings for the classifier used to evaluate generated images.'''

    seed: int
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)


class EvaluationConfig(StrictModel):
    '''Sample sizes and thresholds for the evaluation.'''

    seed: int
    pixel_mse_size: int = Field(ge=1)
    sample_size: int = Field(ge=10)
    confidence_threshold: float = Field(gt=0, lt=1)


class OutputConfig(StrictModel):
    '''Where training runs and reports are written.'''

    run_dir: Path
    report_dir: Path


class ApiConfig(StrictModel):
    '''Inference service settings and request limits.'''

    host: str
    port: int = Field(ge=1, le=65535)
    checkpoint_path: Path
    max_digits: int = Field(ge=1)
    max_variations: int = Field(ge=1)
    max_scale: int = Field(ge=1)


class AppConfig(StrictModel):
    '''Complete project configuration.'''

    device: DeviceName
    data: DataConfig
    training: TrainingConfig
    models: dict[str, GanConfig] = Field(min_length=1)
    classifier: ClassifierConfig
    evaluation: EvaluationConfig
    outputs: OutputConfig
    api: ApiConfig

    def resolve_device(self) -> torch.device:
        '''Return the torch device named in the configuration.'''
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


class EnvironmentSettings(BaseSettings):
    '''Deployment-specific overrides read from the environment or a .env file.'''

    model_config = SettingsConfigDict(env_prefix='DIGITGAN_', env_file='.env', extra='ignore')

    config_path: Path = Path('config/config.yaml')
    log_level: str = 'INFO'
    api_host: str | None = None
    api_port: int | None = None
    checkpoint_path: Path | None = None

    def overrides(self) -> dict[str, dict[str, Any]]:
        '''Return the settings that were provided, grouped by config section.'''
        api = {
            'host': self.api_host,
            'port': self.api_port,
            'checkpoint_path': self.checkpoint_path,
        }
        return {'api': {key: value for key, value in api.items() if value is not None}}


def load_config(
    config_path: Path | None = None, settings: EnvironmentSettings | None = None
) -> AppConfig:
    '''Load the YAML configuration and apply environment overrides.

    Args:
        config_path: YAML file to read. Defaults to the path in the environment settings.
        settings: Environment settings. Read from the process environment when omitted.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    '''
    settings = settings or EnvironmentSettings()
    path = config_path or settings.config_path
    if not path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {path}')

    with path.open(encoding='utf-8') as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    for section, values in settings.overrides().items():
        raw[section] = {**raw.get(section, {}), **values}
    return AppConfig.model_validate(raw)


def configure_logging(level: str = 'INFO', log_format: str = CLI_LOG_FORMAT) -> None:
    '''Configure root logging and quieten chatty third-party loggers.'''
    logging.basicConfig(level=level.upper(), format=log_format, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
