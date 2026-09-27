"""Tests of the attribute classifier and of its labels. Run them with:  python -m pytest tests/"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from models.attribute_classifier import AttributeClassifier, measure_accuracy
from preprocessing.cartoon_dataset import attribute_labels


def test_attribute_labels_follow_the_order_of_mapping():
    words = {'face_color': 'tan', 'hair_color': 'black', 'hair': 'long', 'glasses': 'no glasses',
             'facial_hair': 'a beard'}
    # the position of each word in its list of MAPPING: tan 1 (dark, tan, ...), black 3, long 3, no glasses 2, a beard 0
    assert attribute_labels(words, Config()) == [1, 3, 3, 2, 0]


class FixedAnswers(nn.Module):
    """A fake classifier: always the same logits, whatever the images."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.unused = nn.Parameter(torch.zeros(1))   # measure_accuracy reads the device from the parameters

    def forward(self, images):
        return self.logits


@pytest.mark.parametrize('size', [32, 64])
def test_one_head_per_attribute_at_32_and_64(size):
    logits = AttributeClassifier(Config())(torch.randn(2, 3, size, size))
    assert {attribute: tuple(scores.shape) for attribute, scores in logits.items()} == {
        'face_color': (2, 4), 'hair_color': (2, 5), 'hair': (2, 4), 'glasses': (2, 3), 'facial_hair': (2, 2)}


def test_measure_accuracy_counts_an_image_only_with_all_five_right():
    config = Config()
    labels = torch.tensor([[1, 3, 3, 2, 0],     # tan, black, long, no glasses, a beard
                           [3, 1, 2, 0, 1]])    # pale, ginger, medium, glasses, no beard
    answers = labels.clone()
    answers[1, 2] = 1                            # the second image: short hair instead of medium
    # the highest logit on the answer: 1 there, 0 on the other words
    logits = {attribute: F.one_hot(answers[:, i], len(words)).float()
              for i, (attribute, words) in enumerate(config.MAPPING.items())}
    accuracy = measure_accuracy(FixedAnswers(logits), torch.zeros(2, 3, 32, 32), labels, config)
    assert accuracy['hair'] == 0.5
    assert all(accuracy[attribute] == 1 for attribute in ['face_color', 'hair_color', 'glasses', 'facial_hair'])
    assert accuracy['all'] == 0.5                # one image out of two has all five right
