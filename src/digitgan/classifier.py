'''Convolutional digit classifier used only to evaluate generated images.

The classifier is trained once on the MNIST training split and then frozen. Its penultimate
layer gives the features for FID, and its predictions show whether a generated image is a
recognisable digit and whether it is the digit that was asked for.
'''

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from digitgan.config import ClassifierConfig
from digitgan.data import NUM_CLASSES, DigitImages, make_loader

logger = logging.getLogger(__name__)

CLASSIFIER_FILE = 'classifier.pt'
FEATURE_SIZE = 128


class DigitClassifier(nn.Module):
    '''Two convolutional blocks and a 128-unit feature layer.'''

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, FEATURE_SIZE),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(FEATURE_SIZE, NUM_CLASSES))

    def embed(self, images: Tensor) -> Tensor:
        '''Return the 128 features that FID compares.'''
        features: Tensor = self.body(images)
        return features

    @override
    def forward(self, images: Tensor) -> Tensor:
        logits: Tensor = self.head(self.embed(images))
        return logits


@dataclass(frozen=True)
class ClassifierOutputs:
    '''Features and class probabilities for a batch of images.'''

    features: np.ndarray
    probabilities: np.ndarray


@torch.no_grad()
def classify(
    classifier: DigitClassifier, images: Tensor, batch_size: int = 1000
) -> ClassifierOutputs:
    '''Run the frozen classifier over images in [-1, 1].'''
    device = next(classifier.parameters()).device
    classifier.eval()
    features, probabilities = [], []
    for start in range(0, len(images), batch_size):
        embedded = classifier.embed(images[start : start + batch_size].to(device))
        logits = classifier.head(embedded)
        features.append(embedded.cpu())
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return ClassifierOutputs(
        features=torch.cat(features).double().numpy(),
        probabilities=torch.cat(probabilities).double().numpy(),
    )


def train_classifier(
    train_data: DigitImages, config: ClassifierConfig, device: torch.device
) -> DigitClassifier:
    '''Train the evaluation classifier from scratch.'''
    torch.manual_seed(config.seed)
    classifier = DigitClassifier().to(device)
    optimiser = Adam(classifier.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    loader = make_loader(train_data, config.batch_size, config.seed)

    for epoch in range(1, config.epochs + 1):
        classifier.train()
        total_loss = 0.0
        for images, labels in loader:
            loss = loss_function(classifier(images.to(device)), labels.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        logger.info(
            'Classifier epoch %d/%d: loss %.4f', epoch, config.epochs, total_loss / len(loader)
        )
    return classifier.eval()


def accuracy(classifier: DigitClassifier, data: DigitImages) -> float:
    '''Return the share of images whose most likely class is the true label.'''
    predicted = classify(classifier, data.images).probabilities.argmax(axis=1)
    return float((predicted == data.labels.numpy()).mean())


def save_classifier(classifier: DigitClassifier, test_accuracy: float, path: Path) -> None:
    '''Write the classifier weights and its test accuracy.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': classifier.state_dict(), 'test_accuracy': test_accuracy}, path)


def load_classifier(
    path: Path, device: torch.device | None = None
) -> tuple[DigitClassifier, float]:
    '''Load the classifier and the test accuracy recorded when it was trained.

    Raises:
        FileNotFoundError: If the file does not exist.
    '''
    if not path.is_file():
        raise FileNotFoundError(f'Classifier not found: {path}. Run scripts/train_models.py first.')
    payload: dict[str, Any] = torch.load(path, map_location=device or 'cpu', weights_only=True)
    classifier = DigitClassifier()
    classifier.load_state_dict(payload['state_dict'])
    return classifier.to(device or 'cpu').eval(), float(payload['test_accuracy'])
