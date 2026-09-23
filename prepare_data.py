"""Prepares all the data for training: captions, splits, vocabulary and 32x32 images.

Run it with:  python prepare_data.py
The settings are in config.py, the results go to data/.
"""

import json

import pandas as pd

from config import Config
from preprocessing.caption_generator import CaptionGenerator
from preprocessing.dataset_splitter import DatasetSplitter
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.vocabulary import Vocabulary


def main():
    """Run the steps of the data pipeline and save the settings and the numbers for the report."""
    print('Data preparation for the cartoon diffusion model')
    config = Config()
    config.DATA_DIR.mkdir(exist_ok=True)

    # 1. One caption per image, from its attribute file
    caption_generator = CaptionGenerator(config)
    data = caption_generator.generate_dataset()
    print(f'1. Captions: {len(data)} images')

    # 2. Held-out pairs to ood, then ordinary test, validation and train
    splitter = DatasetSplitter(config)
    ood, in_distribution = splitter.split_ood(data)
    train, val, test = splitter.split_id(in_distribution)
    splitter.check(train, val, test, ood)
    splitter.save(train, val, test, ood)
    print(f'2. Split: train {len(train)}, val {len(val)}, test {len(test)}, ood {len(ood)}')

    # 3. Vocabulary from the training captions only
    vocabulary = Vocabulary(config)
    vocabulary.build(train['caption'])
    vocabulary.check_captions(pd.concat([val, test, ood])['caption'])
    vocabulary.save(config.VOCABULARY_FILE)
    print(f'3. Vocabulary: {len(vocabulary.tokens)} tokens, max_length {vocabulary.max_length}')

    # 4. Images: white background and 32x32, saved as PNG with values 0-255
    #    (the normalization to [-1, 1] happens when CartoonDataset loads them)
    image_preprocessor = ImagePreprocessor(config)
    image_preprocessor.process_dataset(data)
    print(f'4. Images: {len(data)} saved in {config.RESIZED_IMAGES_DIR.name}/')

    # 5. Settings used and numbers for the report
    with open(config.CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)
    summary = splitter.summary(train, val, test, ood)
    summary['vocabulary'] = {'built_from': 'train', 'size': len(vocabulary.tokens),
                             'max_length': vocabulary.max_length}
    with open(config.SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'5. Settings and summary saved in {config.DATA_DIR.name}/')
    for name, count in summary['counts'].items():
        print(f"   {name}: {count['images']} images, {count['combinations']} combinations")
    for name, count in summary['held_out_pairs'].items():
        print(f'   held out {name}: {count} images')
    print(f"   combinations only in train: {len(summary['train_only_combinations'])}")
    print('Done: data ready for the DataLoader.')


if __name__ == '__main__':
    main()
