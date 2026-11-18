'''Tests for the generation API.'''

import io
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from digitgan import __version__
from digitgan.api import create_app
from digitgan.checkpoints import GeneratorCheckpoint, save_checkpoint
from digitgan.config import AppConfig
from digitgan.models import build_generator


@pytest.fixture
def client(
    app_config: AppConfig, conditional_checkpoint: GeneratorCheckpoint
) -> Iterator[TestClient]:
    with TestClient(create_app(app_config, conditional_checkpoint)) as test_client:
        yield test_client


def picture_size(content: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as picture:
        return picture.size


def test_health_reports_the_loaded_generator(client: TestClient) -> None:
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'ok',
        'version': __version__,
        'model_name': 'cgan',
        'training_seed': 1,
    }


def test_generate_returns_the_number_as_a_png(client: TestClient) -> None:
    response = client.get('/generate', params={'digits': '2026', 'scale': 2, 'seed': 7})

    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert response.headers['x-seed'] == '7'
    assert picture_size(response.content) == (4 * 28 * 2, 28 * 2)


def test_each_variation_adds_a_row(client: TestClient) -> None:
    response = client.get('/generate', params={'digits': '12', 'variations': 3, 'scale': 1})

    assert picture_size(response.content) == (2 * 28, 3 * 28)


def test_the_same_seed_gives_the_same_picture(client: TestClient) -> None:
    params: dict[str, str | int] = {'digits': '0123456789', 'seed': 42}

    assert (
        client.get('/generate', params=params).content
        == client.get('/generate', params=params).content
    )


def test_a_random_seed_is_reported_for_reuse(client: TestClient) -> None:
    response = client.get('/generate', params={'digits': '5'})

    assert int(response.headers['x-seed']) >= 0


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'digits': ''},
        {'digits': '12a'},
        {'digits': '1' * 33},
        {'digits': '1', 'variations': 0},
        {'digits': '1', 'variations': 9},
        {'digits': '1', 'scale': 9},
        {'digits': '1', 'seed': -1},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get('/generate', params=params).status_code == 422


def test_root_redirects_to_the_documentation(client: TestClient) -> None:
    response = client.get('/', follow_redirects=False)

    assert response.status_code == 307
    assert response.headers['location'] == '/docs'


def test_configured_checkpoint_is_loaded_at_startup(
    app_config: AppConfig, conditional_checkpoint: GeneratorCheckpoint
) -> None:
    save_checkpoint(conditional_checkpoint, app_config.api.checkpoint_path)

    with TestClient(create_app(app_config)) as client:
        assert client.get('/health').json()['model_name'] == 'cgan'


def test_service_without_a_generator_is_unavailable(app_config: AppConfig) -> None:
    with TestClient(create_app(app_config)) as client:
        assert client.get('/health').status_code == 503
        assert client.get('/generate', params={'digits': '1'}).status_code == 503


def test_unconditional_generators_are_not_served(app_config: AppConfig) -> None:
    unconditional = GeneratorCheckpoint(
        model_name='dcgan',
        display_name='DCGAN',
        architecture='convolutional',
        conditional=False,
        seed=1,
        generator=build_generator('convolutional', conditional=False, latent_dim=100),
        history=(),
    )
    save_checkpoint(unconditional, app_config.api.checkpoint_path)

    with TestClient(create_app(app_config)) as client:
        assert client.get('/health').json()['status'] == 'unavailable'
