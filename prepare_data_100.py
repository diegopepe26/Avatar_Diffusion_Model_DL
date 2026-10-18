"""Same data preparation as prepare_data.py, on all the 100,000 images of cartoonset100k.

The raw data is in dataset/cartoonset100k/100/, the results go to data_100/ (data/ stays as it is).
"""

from config import Config
from prepare_data import main


class Config100(Config):
    """The project Config with the folders of the 100,000 images; all the other settings stay the same."""

    IMAGES_DIR = Config.ROOT / 'dataset' / 'cartoonset100k' / '100' / '100_images'
    ATTRIBUTES_DIR = Config.ROOT / 'dataset' / 'cartoonset100k' / '100' / '100_attributes'

    # every generated file goes to data_100/ instead of data/
    DATA_DIR = Config.ROOT / 'data_100'
    RESIZED_IMAGES_DIR = DATA_DIR / f'images_{Config.IMAGE_SIZE}'
    CAPTIONS_FILE = DATA_DIR / 'captions.csv'
    SPLIT_FILES = {
        'train': DATA_DIR / 'captions_train.csv',
        'val': DATA_DIR / 'captions_val.csv',
        'test': DATA_DIR / 'captions_test.csv',
        'ood': DATA_DIR / 'captions_ood.csv',
    }
    VOCABULARY_FILE = DATA_DIR / 'vocabulary.json'
    CONFIG_FILE = DATA_DIR / 'config.json'
    SUMMARY_FILE = DATA_DIR / 'summary.json'
    PREVIEW_FILE = DATA_DIR / 'preview.png'


if __name__ == '__main__':
    main(Config100())
