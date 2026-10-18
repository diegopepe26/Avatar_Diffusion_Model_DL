"""PyTorch Dataset with the images and captions of one split, ready for training."""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.vocabulary import Vocabulary


def attribute_labels(words, config):
    """Turn the five words of an avatar into the numbers of their classes, the answers of the classifier.

    Args:
        words: {attribute: word}, e.g. {'face_color': 'tan', ...}; a row of the captions table works too.
        config: the project Config (uses MAPPING).

    Returns:
        List of 5 integers in the order of MAPPING, the position of each word in its list,
        e.g. tan, black, long, no glasses, a beard -> [1, 3, 3, 2, 0].
    """
    return [list(groups).index(words[attribute]) for attribute, groups in config.MAPPING.items()]


def attribute_words(numbers, config):
    """Inverse of attribute_labels, from the 5 class numbers back to the words.

    Args:
        numbers: 5 integers in the order of MAPPING, e.g. [1, 3, 3, 2, 0].
        config: the project Config (uses MAPPING).

    Returns:
        {attribute: word}, e.g. {'face_color': 'tan', 'hair_color': 'black', 'hair': 'long', ...}.
    """
    return {attribute: list(groups)[number] for (attribute, groups), number in zip(config.MAPPING.items(), numbers)}


class CartoonDataset(Dataset):
    """Images (normalized to [-1, 1]) and caption ids of one split."""

    def __init__(self, config, split, limit=None, sample=None):
        """Load the table, the vocabulary and the images of one split.

        Args:
            config: the project Config (uses SPLIT_FILES, VOCABULARY_FILE, RESIZED_IMAGES_DIR, MAPPING, SEED).
            split: 'train', 'val', 'test' or 'ood'.
            limit: keep only the first rows of the split (e.g. 64 for the smoke test); None = all of them.
            sample: keep only this many rows, drawn at random with SEED, always the same ones (e.g. to measure a
                100,000-image run on as many images as a 10,000-image one); None = all of them.
        """
        self.table = pd.read_csv(config.SPLIT_FILES[split])
        if limit is not None:
            self.table = self.table.head(limit)    # only these images are read from the disk
        if sample is not None:
            # sort_index keeps the drawn rows in the order of the file. Only these images are read from the disk
            self.table = self.table.sample(n=sample, random_state=config.SEED).sort_index()

        self.vocabulary = Vocabulary(config)
        self.vocabulary.load(config.VOCABULARY_FILE)

        self.images = [ImagePreprocessor.normalize(Image.open(config.RESIZED_IMAGES_DIR / name).convert('RGB'))
                       for name in self.table['file']]
        self.tokens = [torch.tensor(self.vocabulary.encode(caption)) for caption in self.table['caption']]
        self.labels = torch.tensor([attribute_labels(row, config) for _, row in self.table.iterrows()])

    def __len__(self):
        """Number of images in the split."""
        return len(self.table)

    def __getitem__(self, i):
        """Image i as a float tensor (3, IMAGE_SIZE, IMAGE_SIZE) in [-1, 1], and its caption ids (max_length,)."""
        return self.images[i], self.tokens[i]
