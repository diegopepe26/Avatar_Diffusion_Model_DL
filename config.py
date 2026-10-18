"""All the settings of the project in one place (paths, captions, split, images, vocabulary, models)."""

from pathlib import Path


class Config:
    """Settings of the project, read by all the other files."""

    # Every path starts from the project folder
    ROOT = Path(__file__).resolve().parent

    # ---- Raw data, folder 0 of cartoonset100k ----
    IMAGES_DIR = ROOT / 'dataset' / 'cartoonset100k' / '0' / '0_images'
    ATTRIBUTES_DIR = ROOT / 'dataset' / 'cartoonset100k' / '0' / '0_attributes'

    # ---- Images ----
    IMAGE_SIZE = 32                  # size of the images given to the model (32 or 64)
    BACKGROUND = (255, 255, 255)     # color put behind the transparent pixels

    # ---- Generated files ----
    DATA_DIR = ROOT / 'data'
    RESIZED_IMAGES_DIR = DATA_DIR / f'images_{IMAGE_SIZE}'
    CAPTIONS_FILE = DATA_DIR / 'captions.csv'          # all the images
    SPLIT_FILES = {
        'train': DATA_DIR / 'captions_train.csv',
        'val': DATA_DIR / 'captions_val.csv',
        'test': DATA_DIR / 'captions_test.csv',        # ordinary test, seen combinations with new images
        'ood': DATA_DIR / 'captions_ood.csv',          # held-out combinations
    }
    VOCABULARY_FILE = DATA_DIR / 'vocabulary.json'
    CONFIG_FILE = DATA_DIR / 'config.json'             # copy of the settings used to prepare the data
    SUMMARY_FILE = DATA_DIR / 'summary.json'           # numbers of the split and of the vocabulary
    PREVIEW_FILE = DATA_DIR / 'preview.png'            # first 64 resized images

    # ---- Captions ----
    # One template for training captions and generation prompts
    TEMPLATE = ('a cartoon avatar with {face_color} skin, {hair} {hair_color} hair, '
                '{glasses} and {facial_hair}')

    # attribute -> word -> values of the original dataset
    # The word is the exact text that goes into the caption
    MAPPING = {
        'face_color': {'dark': [0, 1, 2], 'tan': [3, 4, 5], 'light': [6, 7], 'pale': [8, 9, 10]},
        'hair_color': {'blonde': [0, 1, 4], 'ginger': [2, 3], 'brown': [5, 6], 'black': [7], 'silver': [8, 9]},
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
    # Combinations of words never seen together in training, which form the OOD set
    OOD_PAIRS = [
        {'face_color': 'dark', 'hair': 'long'},
        {'hair_color': 'blonde', 'glasses': 'sunglasses'},
    ]

    VAL_FRACTION = 0.15      # portion of each combination that goes to validation
    TEST_FRACTION = 0.15     # portion of each combination that goes to the ordinary test
    MIN_IMAGES_TO_SPLIT = 3  # combinations with fewer images stay entirely in train
    SEED = 42

    # ---- Vocabulary ----
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>']   # ids 0, 1, 2, 3

    # ---- Text encoder ----
    D_MODEL = 128             # size of the embeddings, used by the whole model
    NUM_HEADS = 4             # D_MODEL must be divisible by NUM_HEADS
    NUM_ENCODER_BLOCKS = 2    # encoder blocks, one after the other
    FFN_DIM = 4 * D_MODEL     # hidden size of the feed-forward (two linear layers)
    DROPOUT = 0.1

    # ---- Diffusion ----
    NUM_TIMESTEPS = 1000      # T, the noise steps, from t = 0 (almost clean) to t = T - 1 (pure noise)
    # strength of the offset noise, 0 = plain noise; used only at 64x64
    OFFSET_NOISE = 0.1 if IMAGE_SIZE == 64 else 0.0

    # ---- UNet ----
    # channels at 32x32, 16x16, 8x8, doubled at 64x64
    UNET_CHANNELS = (128, 256, 512) if IMAGE_SIZE == 64 else (64, 128, 256)
    TIME_DIM = 4 * UNET_CHANNELS[0]  # size of the time embedding given to every ResBlock
    TEXT_CONDITIONING = True         # False for the unconditional baseline, which reads no text

    # how the caption reaches the UNet at 16x16 and 8x8, either 'cross_attention' (every pixel reads the words it
    # needs) or 'pooled' (the ablation of train_ablation.py, where the mean of the words scales and shifts every
    # channel with FiLM, the same on every pixel)
    CONDITIONING = 'cross_attention'

    # ---- Training ----
    RUNS_DIR = ROOT / 'runs'          # main folder for all experiments

    BATCH_SIZE = 64 if IMAGE_SIZE == 64 else 128  # images per step
    LEARNING_RATE = 2e-4              # AdamW, constant
    GRAD_CLIP = 1.0                   # largest norm of the gradient
    EMA_DECAY = 0.999                 # average of the weights over about the last 1000 steps
    CAPTION_DROPOUT = 0.1             # portion of empty captions, for classifier-free guidance

    EPOCHS = 500                      # 500 for 10k training, 50 for 100k training
    SAMPLE_EVERY = 25                 # epochs between two control grids, 25 for 10k, 5 for 100k
    CHECKPOINT_EVERY = 1              # epochs between two validations + checkpoints
    DETERMINISTIC = False             # True makes two trainings from the same seed give exactly the same model

    GUIDANCE_SCALE = 2.0              # strength of classifier-free guidance when generating
    SAMPLE_PROMPTS = [                # captions of the control grid, 4 seen in training and 4 held-out (OOD)
        'a cartoon avatar with pale skin, short brown hair, glasses and no beard',
        'a cartoon avatar with tan skin, long black hair, no glasses and no beard',
        'a cartoon avatar with light skin, medium ginger hair, sunglasses and a beard',
        'a cartoon avatar with dark skin, short silver hair, no glasses and a beard',
        'a cartoon avatar with dark skin, long black hair, no glasses and no beard',
        'a cartoon avatar with dark skin, long brown hair, glasses and a beard',
        'a cartoon avatar with pale skin, short blonde hair, sunglasses and no beard',
        'a cartoon avatar with tan skin, medium blonde hair, sunglasses and a beard',
    ]

    # ---- Smoke test (train.py --smoke-test) ----
    SMOKE_TEST_IMAGES = 64            # train only on the first images of the training split
    SMOKE_TEST_EPOCHS = 8000          # one step per epoch
    SMOKE_TEST_SAMPLE_EVERY = 1000    # a control grid every 1000 epochs
    SMOKE_TEST_CHECKPOINT_EVERY = 100 # validation + checkpoint every 100 epochs

    # ---- Attribute classifier (train_classifier.py) ----
    CLASSIFIER_CHANNELS = (32, 64, 128)  # channels of the three blocks; the image halves after each block
    CLASSIFIER_EPOCHS = 50               # the best epoch on val is kept
    CLASSIFIER_LEARNING_RATE = 1e-3      # AdamW, constant

    # ---- Evaluation (evaluate.py) ----
    DIVERSITY_IMAGES = 16       # images per SAMPLE_PROMPT for the diversity across seeds
    KID_SUBSETS = 100           # random subsets whose KIDs are averaged
    KID_SUBSET_SIZE = 500       # images in every subset
    DISAGREEMENTS_SHOWN = 32    # generated images with a wrong word shown in the grids

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
