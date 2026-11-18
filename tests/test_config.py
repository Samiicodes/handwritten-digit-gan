'''Tests for loading and validating the configuration.'''

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from digitgan.config import AppConfig, EnvironmentSettings, load_config

from .conftest import CONFIG_PATH


def read_raw_config() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    return raw


def test_repository_config_describes_the_three_gans() -> None:
    config = load_config(CONFIG_PATH, EnvironmentSettings())

    assert list(config.models) == ['vanilla_gan', 'dcgan', 'cgan']
    assert config.models['vanilla_gan'].architecture == 'fully_connected'
    assert config.models['dcgan'].architecture == 'convolutional'
    # The conditional GAN differs from the DCGAN only by conditioning
    assert config.models['cgan'].architecture == config.models['dcgan'].architecture
    assert [model.conditional for model in config.models.values()] == [False, False, True]


def test_unknown_keys_are_rejected() -> None:
    raw = read_raw_config()
    raw['training']['learning_rat'] = 0.1

    with pytest.raises(ValidationError, match='learning_rat'):
        AppConfig.model_validate(raw)


def test_duplicate_seeds_are_rejected() -> None:
    raw = read_raw_config()
    raw['training']['seeds'] = [1, 1]

    with pytest.raises(ValidationError, match='unique'):
        AppConfig.model_validate(raw)


def test_unknown_architecture_is_rejected() -> None:
    raw = read_raw_config()
    raw['models']['dcgan']['architecture'] = 'transformer'

    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_environment_overrides_api_settings() -> None:
    settings = EnvironmentSettings(api_port=9000, checkpoint_path=Path('elsewhere/generator.pt'))

    config = load_config(CONFIG_PATH, settings)

    assert config.api.port == 9000
    assert config.api.checkpoint_path == Path('elsewhere/generator.pt')
    assert config.api.host == '127.0.0.1'


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / 'missing.yaml', EnvironmentSettings())


def test_cpu_device_is_resolved(app_config: AppConfig) -> None:
    assert app_config.resolve_device().type == 'cpu'
