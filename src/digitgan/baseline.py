'''Class-mean baseline: the average training image of each digit.'''

import torch
from torch import Tensor

from digitgan.data import IMAGE_SIZE, NUM_CLASSES, DigitImages


class ClassMeanBaseline:
    '''Returns the mean training image of the requested digit, with no noise and no learning.

    The mean is the single image with the lowest expected squared error against unseen images
    of that digit, so no generator can beat it on pixel MSE. That makes it a useful reference
    for what pixel MSE rewards, as well as the simplest possible conditional generator.
    '''

    def __init__(self, class_means: Tensor) -> None:
        if class_means.shape != (NUM_CLASSES, 1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f'expected class means of shape (10, 1, 28, 28), got {class_means.shape}'
            )
        self.class_means = class_means

    @classmethod
    def fit(cls, data: DigitImages) -> 'ClassMeanBaseline':
        '''Average the training images of each digit.

        Raises:
            ValueError: If a digit has no training images.
        '''
        counts = torch.bincount(data.labels, minlength=NUM_CLASSES)
        if (counts == 0).any():
            raise ValueError('every digit needs at least one training image')
        sums = torch.zeros(NUM_CLASSES, 1, IMAGE_SIZE, IMAGE_SIZE).index_add_(
            0, data.labels, data.images
        )
        return cls(sums / counts.view(-1, 1, 1, 1))

    def generate(self, labels: Tensor) -> Tensor:
        '''Return the mean image of each requested digit.'''
        return self.class_means[labels]
