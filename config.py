"""All the settings of the project in one place: paths, captions, split, images, vocabulary, models."""

from pathlib import Path


class Config:
    """Settings of the project. Change them here, never inside the other files."""

    # Every path starts from the project folder, so the code works on any computer and on Colab.
    ROOT = Path(__file__).resolve().parent

    # ---- Raw data: folder 0 of cartoonset100k (not in git) ----
    IMAGES_DIR = ROOT / 'dataset' / 'cartoonset100k' / '0' / '0_images'
    ATTRIBUTES_DIR = ROOT / 'dataset' / 'cartoonset100k' / '0' / '0_attributes'

    # ---- Images ----
    IMAGE_SIZE = 32                  # side of the images given to the model (32 or 64)
    BACKGROUND = (255, 255, 255)     # color put behind the transparent pixels

    # ---- Generated files (not in git) ----
    DATA_DIR = ROOT / 'data'
    RESIZED_IMAGES_DIR = DATA_DIR / f'images_{IMAGE_SIZE}'
    CAPTIONS_FILE = DATA_DIR / 'captions.csv'          # all the images
    SPLIT_FILES = {
        'train': DATA_DIR / 'captions_train.csv',
        'val': DATA_DIR / 'captions_val.csv',
        'test': DATA_DIR / 'captions_test.csv',        # ordinary test: seen combinations, new images
        'ood': DATA_DIR / 'captions_ood.csv',          # held-out combinations
    }
    VOCABULARY_FILE = DATA_DIR / 'vocabulary.json'
    CONFIG_FILE = DATA_DIR / 'config.json'             # copy of these settings, for the report
    SUMMARY_FILE = DATA_DIR / 'summary.json'           # numbers for the report
    PREVIEW_FILE = DATA_DIR / 'preview.png'            # first 64 resized images

    # ---- Captions ----
    # One template for training captions and generation prompts.
    TEMPLATE = ('a cartoon avatar with {face_color} skin, {hair} {hair_color} hair, '
                '{glasses} and {facial_hair}')

    # attribute -> word -> values of the original dataset.
    # The word is the exact text that goes into the caption.
    MAPPING = {
        'face_color': {'dark': [0, 1, 2], 'tan': [3, 4, 5], 'light': [6, 7], 'pale': [8, 9, 10]},
        'hair_color': {'blonde': [0, 1, 4], 'ginger': [2, 3], 'brown': [5, 6], 'black': [7], 'silver': [8, 9]},
        # Hair length, decided by looking at the 111 hairstyles:
        # balding = bald or almost bald (the hair color stays: it colors eyebrows and beard),
        # short = above the ears (buns included), medium = down to the jaw, long = below the chin.
        'hair': {
            'balding': [0, 1, 2, 108, 109, 110],
            'short': [8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 31, 32, 39, 43, 44, 45, 46, 49, 50,
                      51, 52, 53, 54, 55, 61, 62, 66, 68, 72, 73, 74, 76, 77, 83, 84, 89, 92, 99, 105, 107],
            'medium': [9, 10, 16, 24, 30, 36, 37, 38, 67, 75, 81, 88, 90, 91, 97, 98, 103, 104, 106],
            'long': [3, 4, 5, 6, 7, 13, 15, 25, 26, 27, 28, 29, 33, 34, 35, 40, 41, 42, 47, 48, 56, 57, 58,
                     59, 60, 63, 64, 65, 69, 70, 71, 78, 79, 80, 82, 85, 86, 87, 93, 94, 95, 96, 100, 101, 102],
        },
        'glasses': {'glasses': [2, 3, 4, 5, 6, 7], 'sunglasses': [0, 1, 8, 9, 10], 'no glasses': [11]},
        'facial_hair': {'a beard': list(range(14)), 'no beard': [14]},
    }

    # ---- Split ----
    # Combinations of words never seen together in training: they form the OOD set.
    OOD_PAIRS = [
        {'face_color': 'dark', 'hair': 'long'},
        {'hair_color': 'blonde', 'glasses': 'sunglasses'},
    ]
    VAL_FRACTION = 0.15      # share of each combination that goes to validation
    TEST_FRACTION = 0.15     # share of each combination that goes to the ordinary test
    MIN_IMAGES_TO_SPLIT = 3  # combinations with fewer images stay entirely in train
    SEED = 42

    # ---- Vocabulary ----
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']   # ids 0, 1, 2, 3

    # ---- Text encoder ----
    D_MODEL = 128             # size of the embeddings, used by the whole model (64 or 128)
    NUM_HEADS = 4             # D_MODEL must be divisible by NUM_HEADS
    NUM_ENCODER_BLOCKS = 2    # 2 for now, 4 if needed
    FFN_DIM = 4 * D_MODEL     # hidden size of the feed-forward (two linear layers)
    DROPOUT = 0.1
    ATTENTION = 'scratch'     # 'scratch' (our MultiHeadAttention) or 'torch' (nn.MultiheadAttention)

    # ---- Diffusion ----
    NUM_TIMESTEPS = 1000      # T: noise steps, t = 0 (almost clean) ... T - 1 (pure noise)

    # ---- UNet ----
    UNET_CHANNELS = (64, 128, 256)   # channels at 32x32, 16x16, 8x8
    TIME_DIM = 4 * UNET_CHANNELS[0]  # size of the time embedding given to every ResBlock
    TEXT_CONDITIONING = True         # False: unconditional baseline, no cross-attention

    def to_dict(self):
        """Collect the settings that decide the data, to save them next to it.

        Returns:
            Dictionary ready to be written as data/config.json.
        """
        return {
            'images_dir': str(self.IMAGES_DIR.relative_to(self.ROOT)),
            'attributes_dir': str(self.ATTRIBUTES_DIR.relative_to(self.ROOT)),
            'image_size': self.IMAGE_SIZE,
            'background': self.BACKGROUND,
            'template': self.TEMPLATE,
            'mapping': self.MAPPING,
            'ood_pairs': self.OOD_PAIRS,
            'val_fraction': self.VAL_FRACTION,
            'test_fraction': self.TEST_FRACTION,
            'min_images_to_split': self.MIN_IMAGES_TO_SPLIT,
            'seed': self.SEED,
            'special_tokens': self.SPECIAL_TOKENS,
        }
