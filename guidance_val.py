"""Which guidance to use: the conditional model on the validation captions, at several guidance values.

Run it with:  python guidance_val.py runs/conditional_32     (~35 min per value at 32x32 on an RTX 4060 Laptop)
For every value of GUIDANCES, one after the other, it generates one image per validation caption and
DIVERSITY_IMAGES per seen SAMPLE_PROMPT, always from the same seeds, and measures the conditioning (attribute
classifier), FID and KID against the real validation images, and the diversity. The guidance is chosen here, on
validation: test and OOD are measured once, by evaluate.py. Everything goes to runs/<experiment>/guidance_val/;
stopped and run again, it reads the images already generated instead of generating them again.
"""

import argparse
import json
from pathlib import Path

import torch

from config import Config
from evaluate import check_run, fid_kid, from_pixels, generate_or_load, load_judge, mean_pair_distance, to_pixels
from generate import load_model
from models.attribute_classifier import measure_accuracy
from preprocessing.cartoon_dataset import CartoonDataset
from preprocessing.image_preprocessor import ImagePreprocessor

GUIDANCES = [1.0, 2.0, 3.0, 5.0, 7.0]     # 1 = no push towards the caption; 3 = the value of config.py and evaluate.py


def main(run_dir):
    """Measure every value of GUIDANCES on validation and write runs/<experiment>/guidance_val/guidance_val.json.

    Args:
        run_dir: folder of a conditional experiment, e.g. Path('runs/conditional_32').
    """
    config = Config()
    image_size = check_run(run_dir, config)
    classifier, judge = load_judge(config, image_size)
    model, scheduler, run_config, vocabulary = load_model(run_dir)
    if not run_config.TEXT_CONDITIONING:
        raise ValueError(f'{run_dir.name} has no text: the guidance does nothing there')
    folder = run_dir / 'guidance_val'
    folder.mkdir(exist_ok=True)

    # the real images and captions of the four splits: validation to compare, all of them for the real diversity
    data = {split: CartoonDataset(config, split) for split in ['train', 'val', 'test', 'ood']}
    real_val = to_pixels(torch.stack(data['val'].images))       # 0-255, the pixels of their PNG files
    val_tokens = torch.stack(data['val'].tokens)
    # the diversity on the prompts seen in training only: the held-out ones stay for the final test
    ood_captions = set(data['ood'].table['caption'])
    prompts = [prompt for prompt in config.SAMPLE_PROMPTS if prompt not in ood_captions]
    n = config.DIVERSITY_IMAGES
    prompt_tokens = torch.tensor([vocabulary.encode(prompt) for prompt in prompts for _ in range(n)])
    # the seeds of evaluate.py: validation from SEED, the diversity after one seed per group of validation
    diversity_seed = config.SEED + (len(val_tokens) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    # what the real images give, to read the numbers of the generated ones against
    captions = [caption for dataset in data.values() for caption in dataset.table['caption']]
    all_real = torch.cat([torch.stack(dataset.images) for dataset in data.values()])    # in the order of captions
    real_diversity = sum(mean_pair_distance(all_real[torch.tensor([caption == prompt for caption in captions])])
                         for prompt in prompts) / len(prompts)
    reference = fid_kid(real_val, to_pixels(torch.stack(data['test'].images)), config)      # real val vs real test
    print(f'Real images: all five words right in {judge["val"]["all"]:.1%} of val, FID val-test '
          f'{reference["fid"]:.2f}, diversity {real_diversity:.3f}')

    rows = []
    for guidance in GUIDANCES:
        print(f'Guidance {guidance:g}')
        run_config.GUIDANCE_SCALE = guidance     # read by sample(): only this copy of the settings, in memory
        generated = generate_or_load(folder / f'generated_val_guidance{guidance:g}.pt', model, scheduler, run_config,
                                     val_tokens, config.SEED)
        diverse = generate_or_load(folder / f'generated_diversity_guidance{guidance:g}.pt', model, scheduler,
                                   run_config, prompt_tokens, diversity_seed)
        ImagePreprocessor(run_config).save_preview(
            [ImagePreprocessor.denormalize(image) for image in from_pixels(diverse['images'])],
            folder / f'diversity_guidance{guidance:g}.png')
        row = {'guidance': guidance,
               'conditioning': measure_accuracy(classifier, from_pixels(generated['images']), data['val'].labels,
                                                config),
               'quality': fid_kid(real_val, generated['images'], config),
               'diversity': sum(mean_pair_distance(from_pixels(diverse['images'][k * n:(k + 1) * n]))
                                for k in range(len(prompts))) / len(prompts)}
        rows.append(row)
        print(f'   all five right {row["conditioning"]["all"]:.1%}, FID {row["quality"]["fid"]:.2f}, '
              f'KID {row["quality"]["kid_mean"]:.4f}, diversity {row["diversity"]:.3f}')

    results = {'experiment': run_dir.name, 'split': 'val', 'images': len(val_tokens), 'diversity_prompts': prompts,
               'real': {'all_five_right': judge['val']['all'], 'fid_val_test': reference, 'diversity': real_diversity},
               'guidances': rows}
    with open(folder / 'guidance_val.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n{"guidance":>9} {"all five right":>15} {"FID":>7} {"KID":>8} {"diversity":>10}')
    for row in rows:
        print(f'{row["guidance"]:>9g} {row["conditioning"]["all"]:>15.1%} {row["quality"]["fid"]:>7.2f} '
              f'{row["quality"]["kid_mean"]:>8.4f} {row["diversity"]:>10.3f}')
    print(f'{"real":>9} {judge["val"]["all"]:>15.1%} {reference["fid"]:>7.2f} {reference["kid_mean"]:>8.4f} '
          f'{real_diversity:>10.3f}')
    print(f'Done: {folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure several guidance values on the validation captions.')
    parser.add_argument('run_dir', type=Path, help='folder of a conditional experiment, e.g. runs/conditional_32')
    main(parser.parse_args().run_dir)
