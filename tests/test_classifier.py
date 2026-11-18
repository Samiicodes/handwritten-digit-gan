'''Tests for the evaluation classifier.'''

from pathlib import Path

import numpy as np
import pytest

from digitgan.classifier import (
    FEATURE_SIZE,
    DigitClassifier,
    accuracy,
    classify,
    load_classifier,
    save_classifier,
)
from digitgan.config import AppConfig
from digitgan.data import DigitImages


def test_classifier_learns_the_synthetic_digits(
    trained_classifier: DigitClassifier, test_data: DigitImages
) -> None:
    assert accuracy(trained_classifier, test_data) > 0.9


def test_classify_returns_features_and_probabilities(
    trained_classifier: DigitClassifier, test_data: DigitImages
) -> None:
    outputs = classify(trained_classifier, test_data.images, batch_size=64)

    assert outputs.features.shape == (len(test_data), FEATURE_SIZE)
    assert outputs.probabilities.shape == (len(test_data), 10)
    assert np.allclose(outputs.probabilities.sum(axis=1), 1)


def test_classifier_round_trips_through_a_file(
    trained_classifier: DigitClassifier, test_data: DigitImages, tmp_path: Path
) -> None:
    path = tmp_path / 'classifier.pt'

    save_classifier(trained_classifier, 0.97, path)
    loaded, recorded_accuracy = load_classifier(path)

    assert recorded_accuracy == 0.97
    assert np.allclose(
        classify(loaded, test_data.images).probabilities,
        classify(trained_classifier, test_data.images).probabilities,
    )


def test_loading_a_missing_classifier_explains_what_to_run(app_config: AppConfig) -> None:
    with pytest.raises(FileNotFoundError, match=r'train_models\.py'):
        load_classifier(app_config.outputs.run_dir / 'classifier.pt')
