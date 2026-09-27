"""Tests of the attribute classifier and of its labels. Run them with:  python -m pytest tests/"""

from config import Config
from preprocessing.cartoon_dataset import attribute_labels


def test_attribute_labels_follow_the_order_of_mapping():
    words = {'face_color': 'tan', 'hair_color': 'black', 'hair': 'long', 'glasses': 'no glasses',
             'facial_hair': 'a beard'}
    # the position of each word in its list of MAPPING: tan 1 (dark, tan, ...), black 3, long 3, no glasses 2, a beard 0
    assert attribute_labels(words, Config()) == [1, 3, 3, 2, 0]
