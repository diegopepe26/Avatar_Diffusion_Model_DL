"""Attribute classifier: a small CNN that tells the five words of an avatar (skin, hair color, hair, glasses, beard).

It is the judge of the conditioning metric: trained from scratch on the real training images and on half of the
real OOD images (the diffusion model never sees them), then frozen, it reads the generated images and says
whether they show the words of their prompt.
"""

import torch
from torch import nn


class AttributeClassifier(nn.Module):
    """Shared convolutions, then one linear head per attribute of MAPPING."""

    def __init__(self, config):
        """Args:
            config: the project Config (uses CLASSIFIER_CHANNELS, MAPPING).
        """
        super().__init__()
        layers = []
        in_channels = 3
        for channels in config.CLASSIFIER_CHANNELS:
            # two 3x3 convolutions, then the image becomes half as big: 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(in_channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
                       nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
                       nn.MaxPool2d(2)]
            in_channels = channels
        self.features = nn.Sequential(*layers)
        # one output per word: face_color 4, hair_color 5, hair 4, glasses 3, facial_hair 2
        self.heads = nn.ModuleDict({attribute: nn.Linear(in_channels, len(words))
                                    for attribute, words in config.MAPPING.items()})

    def forward(self, images):
        """Args:
            images: (B, 3, H, W) in [-1, 1]: B = images in the batch, 3 = RGB colors,
                H = W = side of the images (IMAGE_SIZE: 32 or 64).

        Returns:
            {attribute: (B, number of words) logits}, in the order of MAPPING: one score per word for every image,
            e.g. logits['hair'] is (B, 4), the scores of balding, short, medium and long.
        """
        # the mean over all the pixels: the same number of values at 32x32 and at 64x64
        features = self.features(images).mean(dim=(2, 3))
        return {attribute: head(features) for attribute, head in self.heads.items()}


def measure_accuracy(classifier, images, labels, config):
    """Share of right answers of the classifier, for each attribute and for all five together.

    Used on the real images (train_classifier.py) and on the generated ones (evaluate.py): the two are
    measured in exactly the same way.

    Args:
        classifier: AttributeClassifier (it is left in eval()).
        images: (N, 3, H, W) in [-1, 1], on any device: N = number of images (e.g. 1255 for val),
            3 = RGB colors, H = W = side of the images (IMAGE_SIZE).
        labels: (N, 5) the right class numbers of every image, one per attribute in the order of MAPPING
            (see attribute_labels), e.g. [1, 3, 3, 2, 0] = tan, black, long, no glasses, a beard.
        config: the project Config (uses BATCH_SIZE, MAPPING).

    Returns:
        {attribute: fraction of right answers, ..., 'all': fraction of images with all five right}.
    """
    classifier.eval()     # BatchNorm with the statistics of the real training images
    device = next(classifier.parameters()).device
    right = []            # one (images in the batch, 5) tensor per batch: True where the word is right
    with torch.no_grad():
        for start in range(0, len(images), config.BATCH_SIZE):
            logits = classifier(images[start:start + config.BATCH_SIZE].to(device))
            # the word with the highest score, for each attribute: (images in the batch, 5)
            answers = torch.stack([logits[attribute].argmax(dim=1) for attribute in config.MAPPING], dim=1)
            right.append(answers.cpu() == labels[start:start + config.BATCH_SIZE].cpu())
    right = torch.cat(right)
    accuracy = {attribute: right[:, i].float().mean().item() for i, attribute in enumerate(config.MAPPING)}
    accuracy['all'] = right.all(dim=1).float().mean().item()
    return accuracy
