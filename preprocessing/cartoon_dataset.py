"""PyTorch Dataset: the images and captions of one split, ready for training."""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.vocabulary import Vocabulary


class CartoonDataset(Dataset):
    """Images (normalized to [-1, 1]) and caption ids of one split."""

    def __init__(self, config, split):
        """Load the table, the vocabulary and all the images of one split.

        Args:
            config: the project Config (uses SPLIT_FILES, VOCABULARY_FILE, RESIZED_IMAGES_DIR).
            split: 'train', 'val', 'test' or 'ood'.
        """
        if split not in config.SPLIT_FILES:
            raise ValueError(f"split must be one of {list(config.SPLIT_FILES)}, not '{split}'")

        # keep_default_na=False: no word is ever read as a missing value
        self.table = pd.read_csv(config.SPLIT_FILES[split], keep_default_na=False)
        self.vocabulary = Vocabulary(config)
        self.vocabulary.load(config.VOCABULARY_FILE)

        # everything in memory, ready to use: the epochs do not read the disk
        self.images = [ImagePreprocessor.normalize(Image.open(config.RESIZED_IMAGES_DIR / name).convert('RGB'))
                       for name in self.table['file']]
        self.tokens = [torch.tensor(self.vocabulary.encode(caption)) for caption in self.table['caption']]

    def __len__(self):
        """Returns: number of images in the split."""
        return len(self.table)

    def __getitem__(self, i):
        """Returns: (image float (3, 32, 32) in [-1, 1], caption ids (max_length,))."""
        return self.images[i], self.tokens[i]
