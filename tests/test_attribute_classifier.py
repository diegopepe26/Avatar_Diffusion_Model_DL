"""Tests of the attribute classifier and of its labels. Run them with:  python -m pytest tests/"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from models.attribute_classifier import AttributeClassifier, load_classifier, measure_accuracy
from preprocessing.cartoon_dataset import attribute_labels, attribute_words


def test_attribute_labels_follow_the_order_of_mapping():
    words = {'face_color': 'tan', 'hair_color': 'black', 'hair': 'long', 'glasses': 'no glasses',
             'facial_hair': 'a beard'}
    # the position of each word in its list of MAPPING: tan 1 (dark, tan, ...), black 3, long 3, no glasses 2, a beard 0
    assert attribute_labels(words, Config()) == [1, 3, 3, 2, 0]
    assert attribute_words([1, 3, 3, 2, 0], Config()) == words      # and back


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


def test_load_classifier_rebuilds_the_saved_network(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'RUNS_DIR', tmp_path)          # instead of runs/
    assert load_classifier(Config(), 32) is None               # not trained yet at this size
    config = Config()
    config.CLASSIFIER_CHANNELS = (8, 16, 32)                   # not the channels of config.py
    saved = AttributeClassifier(config)
    (tmp_path / 'classifier_32').mkdir()
    torch.save({'weights': saved.state_dict(), 'settings': {'IMAGE_SIZE': 32, 'CLASSIFIER_CHANNELS': (8, 16, 32)},
                'epoch': 1, 'val_accuracy': 0.5}, tmp_path / 'classifier_32' / 'classifier.pt')
    loaded = load_classifier(Config(), 32)
    assert loaded.heads['hair'].in_features == 32              # rebuilt with the saved channels
    assert torch.equal(loaded.heads['hair'].weight.cpu(), saved.heads['hair'].weight)
    assert not loaded.training                                 # ready to judge: eval()
