'''Generator and discriminator architectures for the vanilla GAN, DCGAN, and conditional GAN.

Every network supports conditioning, so the conditional GAN is the DCGAN with exactly one
change: both networks are told which digit is wanted.
'''

from typing import override

import torch
from torch import Tensor, nn

from digitgan.config import Architecture
from digitgan.data import IMAGE_SIZE, NUM_CLASSES

PIXELS = IMAGE_SIZE * IMAGE_SIZE
# Two stride-2 transposed convolutions grow 7x7 feature maps into 28x28 images
_BASE_SIZE = IMAGE_SIZE // 4
_BASE_CHANNELS = 128
_LEAK = 0.2


class Generator(nn.Module):
    '''Maps noise, plus a digit label for conditional models, to images in [-1, 1].'''

    def __init__(self, latent_dim: int, conditional: bool) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.label_embedding = nn.Embedding(NUM_CLASSES, NUM_CLASSES) if conditional else None
        self.input_size = latent_dim + (NUM_CLASSES if conditional else 0)

    def join_label(self, noise: Tensor, labels: Tensor | None) -> Tensor:
        '''Append the embedded label to the noise vector, which is how conditioning enters.'''
        if self.label_embedding is None:
            if labels is not None:
                raise ValueError('this generator is unconditional and cannot take labels')
            return noise
        if labels is None:
            raise ValueError('a conditional generator needs labels')
        return torch.cat([noise, self.label_embedding(labels)], dim=1)

    @override
    def forward(self, noise: Tensor, labels: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class Discriminator(nn.Module):
    '''Scores images as real or generated. Returns one logit per image.'''

    def __init__(self, conditional: bool) -> None:
        super().__init__()
        self.conditional = conditional
        # The label becomes a learned 28x28 map stacked with the image as a second channel
        self.label_embedding = nn.Embedding(NUM_CLASSES, PIXELS) if conditional else None
        self.input_channels = 2 if conditional else 1

    def stack_label(self, images: Tensor, labels: Tensor | None) -> Tensor:
        '''Add the label map as an extra channel for conditional models.'''
        if self.label_embedding is None:
            if labels is not None:
                raise ValueError('this discriminator is unconditional and cannot take labels')
            return images
        if labels is None:
            raise ValueError('a conditional discriminator needs labels')
        label_maps = self.label_embedding(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return torch.cat([images, label_maps], dim=1)

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class FullyConnectedGenerator(Generator):
    '''Vanilla GAN generator: a multilayer perceptron that outputs all 784 pixels at once.'''

    def __init__(self, latent_dim: int, conditional: bool) -> None:
        super().__init__(latent_dim, conditional)
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(_LEAK),
            nn.Linear(256, 512),
            nn.LeakyReLU(_LEAK),
            nn.Linear(512, 1024),
            nn.LeakyReLU(_LEAK),
            nn.Linear(1024, PIXELS),
            nn.Tanh(),
        )

    @override
    def forward(self, noise: Tensor, labels: Tensor | None = None) -> Tensor:
        pixels: Tensor = self.layers(self.join_label(noise, labels))
        return pixels.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


class FullyConnectedDiscriminator(Discriminator):
    '''Vanilla GAN discriminator: flattens the image, so it has no notion of neighbouring pixels.'''

    def __init__(self, conditional: bool) -> None:
        super().__init__(conditional)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_channels * PIXELS, 1024),
            nn.LeakyReLU(_LEAK),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(_LEAK),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(_LEAK),
            nn.Linear(256, 1),
        )

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        logits: Tensor = self.layers(self.stack_label(images, labels))
        return logits.squeeze(1)


class ConvolutionalGenerator(Generator):
    '''DCGAN generator: projects noise to 7x7 feature maps and upsamples them to 28x28.'''

    def __init__(self, latent_dim: int, conditional: bool) -> None:
        super().__init__(latent_dim, conditional)
        self.project = nn.Linear(self.input_size, _BASE_CHANNELS * _BASE_SIZE * _BASE_SIZE)
        self.upsample = nn.Sequential(
            nn.BatchNorm2d(_BASE_CHANNELS),
            nn.LeakyReLU(_LEAK),
            nn.ConvTranspose2d(_BASE_CHANNELS, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(_LEAK),
            # Padding 1 gives 28x28. The coursework used 3 here, which gave 24x24 images.
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    @override
    def forward(self, noise: Tensor, labels: Tensor | None = None) -> Tensor:
        features: Tensor = self.project(self.join_label(noise, labels))
        images: Tensor = self.upsample(features.view(-1, _BASE_CHANNELS, _BASE_SIZE, _BASE_SIZE))
        return images


class ConvolutionalDiscriminator(Discriminator):
    '''DCGAN discriminator: strided convolutions downsample 28x28 to 7x7 before the decision.'''

    def __init__(self, conditional: bool) -> None:
        super().__init__(conditional)
        self.layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(_LEAK),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, _BASE_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(_BASE_CHANNELS),
            nn.LeakyReLU(_LEAK),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(_BASE_CHANNELS * _BASE_SIZE * _BASE_SIZE, 1),
        )

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        logits: Tensor = self.layers(self.stack_label(images, labels))
        return logits.squeeze(1)


def build_generator(architecture: Architecture, conditional: bool, latent_dim: int) -> Generator:
    '''Create an untrained generator.'''
    if architecture == 'fully_connected':
        return FullyConnectedGenerator(latent_dim, conditional)
    return ConvolutionalGenerator(latent_dim, conditional)


def build_discriminator(architecture: Architecture, conditional: bool) -> Discriminator:
    '''Create an untrained discriminator.'''
    if architecture == 'fully_connected':
        return FullyConnectedDiscriminator(conditional)
    return ConvolutionalDiscriminator(conditional)


def count_parameters(module: nn.Module) -> int:
    '''Return the number of trainable parameters.'''
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
