'''Tests for the class-mean baseline.'''

import pytest
import torch

from digitgan.baseline import ClassMeanBaseline
from digitgan.data import DigitImages


def test_fit_averages_the_images_of_each_digit(train_data: DigitImages) -> None:
    baseline = ClassMeanBaseline.fit(train_data)

    expected = train_data.images[train_data.labels == 4].mean(dim=0)
    assert torch.allclose(baseline.class_means[4], expected, atol=1e-6)


def test_generate_returns_the_mean_of_each_requested_digit(train_data: DigitImages) -> None:
    baseline = ClassMeanBaseline.fit(train_data)

    images = baseline.generate(torch.tensor([2, 2, 9]))

    assert images.shape == (3, 1, 28, 28)
    assert torch.equal(images[0], images[1])
    assert torch.equal(images[2], baseline.class_means[9])


def test_fit_needs_every_digit() -> None:
    data = DigitImages(torch.zeros(3, 1, 28, 28), torch.tensor([0, 1, 1]))

    with pytest.raises(ValueError, match='every digit'):
        ClassMeanBaseline.fit(data)


def test_class_means_must_cover_ten_digits() -> None:
    with pytest.raises(ValueError, match='shape'):
        ClassMeanBaseline(torch.zeros(9, 1, 28, 28))
