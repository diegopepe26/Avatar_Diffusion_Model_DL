"""Evaluation of a trained experiment on the captions of the ordinary test and of the held-out combinations (OOD).

Run it with:  python evaluate.py runs/conditional_32     (after train.py and train_classifier.py)
It generates one image per row of the test and OOD splits and measures the conditioning (with the attribute
classifier), the quality (FID and KID, with real-vs-real references), the diversity across seeds (SAMPLE_PROMPTS),
the parameters, the sampling time and the GPU memory. Everything goes to runs/<experiment>/evaluation/.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from config import Config
from generate import load_model
from models.attribute_classifier import load_classifier, measure_accuracy, predict
from models.sampling import sample
from preprocessing.cartoon_dataset import CartoonDataset, attribute_words
from preprocessing.image_preprocessor import ImagePreprocessor


def to_pixels(images):
    """From the scale of the model to the scale of the PNG files, as the real images.

    Args:
        images: (N, 3, H, W) float in [-1, 1]: N = number of images, 3 = RGB colors, H = W = side of the images.

    Returns:
        (N, 3, H, W) uint8 on the CPU, values 0-255.
    """
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def from_pixels(pixels):
    """Inverse of to_pixels: from 0-255 back to the scale of the model and of the classifier.

    Args:
        pixels: (N, 3, H, W) uint8: N = number of images, 3 = RGB colors, H = W = side of the images.

    Returns:
        (N, 3, H, W) float in [-1, 1].
    """
    return pixels.float() / 127.5 - 1


def check_run(run_dir, config):
    """Stop before the long generation if the experiment cannot be evaluated.

    Args:
        run_dir: folder of the experiment, e.g. runs/conditional_32.
        config: the project Config (uses IMAGE_SIZE: the real images are read from data/images_<IMAGE_SIZE>).

    Returns:
        The side of the images of the experiment (32 or 64).
    """
    checkpoint_file = run_dir / 'last.pt'
    if not checkpoint_file.exists():
        raise ValueError(f'No last.pt in {run_dir}: train the experiment first with train.py')
    image_size = torch.load(checkpoint_file, map_location='cpu', weights_only=False)['settings']['IMAGE_SIZE']
    # the generated images are compared with the real ones of data/images_<IMAGE_SIZE of config.py>
    if image_size != config.IMAGE_SIZE:
        raise ValueError(f'{run_dir.name} generates {image_size}x{image_size} images but config.py has '
                         f'IMAGE_SIZE = {config.IMAGE_SIZE}: set IMAGE_SIZE = {image_size} in config.py')
    return image_size


def load_judge(config, image_size):
    """The attribute classifier of the size of the experiment, and how often it is right on the real images.

    Args:
        config: the project Config (uses RUNS_DIR).
        image_size: side of the generated images (32 or 64).

    Returns:
        (AttributeClassifier in eval() on the GPU if there is one, the content of its accuracy.json).
    """
    classifier = load_classifier(config, image_size)
    if classifier is None:
        raise ValueError(f'No attribute classifier for {image_size}x{image_size}: '
                         f'run train_classifier.py with IMAGE_SIZE = {image_size}')
    folder = config.RUNS_DIR / f'classifier_{image_size}'
    # an interrupted training overwrites classifier.pt at its first epoch but writes accuracy.json only at the end
    epoch = torch.load(folder / 'classifier.pt', map_location='cpu')['epoch']
    report = json.loads((folder / 'accuracy.json').read_text()) if (folder / 'accuracy.json').exists() else {}
    if report.get('epoch') != epoch:
        raise ValueError(f'{folder}: accuracy.json is not of the training of classifier.pt (epoch {epoch}): '
                         'run train_classifier.py again')
    return classifier, report


def generate_images(model, scheduler, config, tokens, first_seed):
    """Generate one image per caption, in groups of BATCH_SIZE images, each group with its own seed.

    Args:
        model, scheduler: as returned by load_model (the model in eval() on its device).
        config: the Config of the experiment (uses BATCH_SIZE, IMAGE_SIZE, GUIDANCE_SCALE).
        tokens: (N, L) caption ids: N = number of images, L = tokens per caption.
        first_seed: seed of the first group; the next groups get first_seed + 1, + 2, ...

    Returns:
        (N, 3, IMAGE_SIZE, IMAGE_SIZE) uint8 images on the CPU, in the order of the captions.
    """
    device = next(model.parameters()).device
    groups = (len(tokens) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    images = []
    for number, start in enumerate(range(0, len(tokens), config.BATCH_SIZE)):
        print(f'   group {number + 1}/{groups}')
        # a seed per group: with the same seed, the images in the same position of two groups would start from
        # the same noise and look alike in everything the caption does not say
        group = sample(model, scheduler, tokens[start:start + config.BATCH_SIZE].to(device), config,
                       first_seed + number)
        images.append(to_pixels(group))
    return torch.cat(images)


def generate_or_load(path, model, scheduler, config, tokens, first_seed):
    """Read the images of a previous run if they are saved, otherwise generate them and save them.

    A crash after the generation (e.g. during the FID) or a fixed metric then costs minutes, not a new generation.

    Args:
        path: the .pt file, e.g. runs/conditional_32/evaluation/generated_test.pt.
        model, scheduler, config, tokens, first_seed: as in generate_images (not used if the file exists).

    Returns:
        {'images': (N, 3, H, W) uint8, 'seconds': time of the generation, 'peak_memory_mb': highest GPU memory
        during it (None without a GPU)}: N = number of images, H = W = side of the images.
    """
    if path.exists():
        print(f'   {path.name} already exists: read from the disk, not generated again')
        return torch.load(path)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()     # the peak of this generation only
    start = time.time()
    images = generate_images(model, scheduler, config, tokens, first_seed)
    result = {'images': images, 'seconds': time.time() - start,
              'peak_memory_mb': torch.cuda.max_memory_allocated() / 2**20 if cuda else None}
    torch.save(result, path)
    return result


def mean_pair_distance(images):
    """Diversity of a group of images: the mean distance over all the pairs of different images.

    The distance of two images is the mean of the absolute differences of their values, with the pixels in [0, 1]:
    0 for two identical images, 1 between a black and a white one. N images make N * (N - 1) / 2 pairs.

    Args:
        images: (N, 3, H, W) float in [-1, 1]: N = number of images (at least 2), 3 = RGB colors,
            H = W = side of the images.

    Returns:
        The mean distance over the pairs, a float.
    """
    pixels = ((images.float() + 1) / 2).flatten(1)                      # (N, 3 * H * W) in [0, 1]
    # distance of every image with every other: (N, N), 0 on the diagonal (an image with itself)
    distances = torch.cdist(pixels, pixels, p=1) / pixels.shape[1]
    first, second = torch.triu_indices(len(pixels), len(pixels), offset=1)   # every pair once: first < second
    return distances[first, second].mean().item()


def disagreement_line(position, caption, asked, seen):
    """One line of the disagreements file: the caption and the words the classifier sees differently.

    Args:
        position: number of the image in the grid (1 = top left, then left to right, top to bottom).
        caption: the prompt of the image.
        asked: {attribute: word} of the caption, in the order of MAPPING.
        seen: {attribute: word} the classifier sees in the image.

    Returns:
        e.g. '5. a cartoon avatar with ... — hair: asked long, sees medium'.
    """
    wrong = [f'{attribute}: asked {asked[attribute]}, sees {seen[attribute]}'
             for attribute in asked if seen[attribute] != asked[attribute]]
    return f'{position}. {caption} — ' + '; '.join(wrong)


def fid_kid(real, generated, config):
    """FID and KID between two groups of images, on the 2048 Inception-v3 features (torchmetrics, torch-fidelity).

    Args:
        real, generated: (N, 3, H, W) float in [-1, 1]: N = number of images of each group (the two may differ),
            3 = RGB colors, H = W = side of the images (resized to 299x299 inside the library).
        config: the project Config (uses KID_SUBSETS, KID_SUBSET_SIZE, BATCH_SIZE, SEED).

    Returns:
        {'fid': float, 'kid_mean': float, 'kid_std': float, 'images': [real N, generated N]}.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # normalize=True: the images are given as floats in [0, 1]
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=2048, subsets=config.KID_SUBSETS, subset_size=config.KID_SUBSET_SIZE,
                                  normalize=True).to(device)
    for images, is_real in [(real, True), (generated, False)]:
        for start in range(0, len(images), config.BATCH_SIZE):
            group = ((images[start:start + config.BATCH_SIZE].float() + 1) / 2).to(device)
            fid.update(group, real=is_real)
            kid.update(group, real=is_real)
    torch.manual_seed(config.SEED)       # the random subsets of the KID
    kid_mean, kid_std = kid.compute()
    return {'fid': fid.compute().item(), 'kid_mean': kid_mean.item(), 'kid_std': kid_std.item(),
            'images': [len(real), len(generated)]}


def measure_one_image(model, scheduler, config, tokens):
    """Time and GPU memory of one image generated alone, as in the demo.

    Args:
        model, scheduler: as returned by load_model.
        config: the Config of the experiment (uses SEED, IMAGE_SIZE, GUIDANCE_SCALE).
        tokens: (1, L) ids of one caption: L = tokens per caption.

    Returns:
        (seconds, highest GPU memory in MB or None without a GPU).
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    sample(model, scheduler, tokens.to(next(model.parameters()).device), config, config.SEED)
    if cuda:
        torch.cuda.synchronize()     # the GPU works in the background: wait for it before reading the clock
    return time.time() - start, torch.cuda.max_memory_allocated() / 2**20 if cuda else None


def save_disagreements(images, labels, captions, classifier, config, folder, split):
    """Save the grid and the list of the first generated images with a word the classifier sees differently.

    Args:
        images: (N, 3, H, W) uint8 generated images: N = images of the split, H = W = IMAGE_SIZE.
        labels: (N, 5) class numbers of their captions, in the order of MAPPING.
        captions: the N captions.
        classifier: AttributeClassifier of this image size.
        config: the Config of the experiment (uses DISAGREEMENTS_SHOWN, MAPPING, BATCH_SIZE, IMAGE_SIZE).
        folder: the evaluation folder.
        split: 'test' or 'ood', in the file names.
    """
    answers = predict(classifier, from_pixels(images), config)
    wrong = (answers != labels).any(dim=1).nonzero().flatten()[:config.DISAGREEMENTS_SHOWN].tolist()
    lines = [disagreement_line(position, captions[i], attribute_words(labels[i].tolist(), config),
                               attribute_words(answers[i].tolist(), config))
             for position, i in enumerate(wrong, start=1)]
    with open(folder / f'disagreements_{split}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n' if lines else 'No generated image with a wrong word.\n')
    grid_file = folder / f'disagreements_{split}.png'
    grid_file.unlink(missing_ok=True)     # the grid of a previous run must not stay when there is nothing wrong
    if wrong:
        ImagePreprocessor(config).save_preview([ImagePreprocessor.denormalize(from_pixels(images[i])) for i in wrong],
                                               grid_file)


def main(run_dir):
    """Evaluate one experiment and write everything to runs/<experiment>/evaluation/.

    Args:
        run_dir: folder of the experiment, e.g. Path('runs/conditional_32').
    """
    config = Config()
    print(f'Evaluation of {run_dir}')

    # 1. The checks first: they stop the run before the long generation
    image_size = check_run(run_dir, config)
    classifier, judge = load_judge(config, image_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('WARNING: no GPU found, the generation on CPU will take many hours (see README, Setup)')
    model, scheduler, run_config, vocabulary = load_model(run_dir)
    folder = run_dir / 'evaluation'
    folder.mkdir(exist_ok=True)
    print(f'1. {run_dir.name}: {image_size}x{image_size}, {"with" if run_config.TEXT_CONDITIONING else "without"} '
          f'text, guidance {run_config.GUIDANCE_SCALE}; judge: epoch {judge["epoch"]} (device: {device})')

    # 2. The real images, captions and labels of the four splits
    data = {split: CartoonDataset(config, split) for split in ['train', 'val', 'test', 'ood']}
    real = {split: torch.stack(dataset.images) for split, dataset in data.items()}
    print('2. Data: ' + ', '.join(f'{split} {len(dataset)}' for split, dataset in data.items()))

    # 3. One image per row of test and OOD, then the diversity group; the seeds of the groups one after the other
    seed = config.SEED
    generated = {}
    for split in ['test', 'ood']:
        tokens = torch.stack(data[split].tokens)
        print(f'3. Generation of {split}: {len(tokens)} images, first seed {seed}')
        generated[split] = generate_or_load(folder / f'generated_{split}.pt', model, scheduler, run_config, tokens,
                                            seed)
        seed += (len(tokens) + config.BATCH_SIZE - 1) // config.BATCH_SIZE     # one seed per group
    prompts = config.SAMPLE_PROMPTS
    tokens = torch.tensor([vocabulary.encode(prompt) for prompt in prompts for _ in range(config.DIVERSITY_IMAGES)])
    print(f'   diversity: {len(tokens)} images, seed {seed}')
    diverse = generate_or_load(folder / 'generated_diversity.pt', model, scheduler, run_config, tokens, seed)
    # one image alone, as in the demo: measured now, before the Inception networks of FID and KID take GPU memory
    single_seconds, single_memory = measure_one_image(model, scheduler, run_config, data['test'].tokens[0][None])

    # 4. Conditioning: the words the classifier sees in the generated images, next to its accuracy on the real ones
    conditioning = {'judge': {'epoch': judge['epoch'], 'val_accuracy': judge['val']['all']}}
    for split in ['test', 'ood']:
        result = measure_accuracy(classifier, from_pixels(generated[split]['images']), data[split].labels, config)
        result['images'] = len(data[split])
        conditioning[split] = {'generated': result, 'real': judge[split]}
        save_disagreements(generated[split]['images'], data[split].labels, list(data[split].table['caption']),
                           classifier, run_config, folder, split)
    conditioning['ood_pairs'] = {}
    for pair in config.OOD_PAIRS:
        name = ' + '.join(pair.values())     # as in the column ood_pair, e.g. 'dark + long'
        # contains, not ==: an image with both pairs has 'dark + long; blonde + sunglasses' and counts in both
        rows = torch.tensor(data['ood'].table['ood_pair'].str.contains(name, regex=False).to_numpy())
        result = measure_accuracy(classifier, from_pixels(generated['ood']['images'][rows]), data['ood'].labels[rows],
                                  config)
        result['images'] = int(rows.sum())
        conditioning['ood_pairs'][name] = {'generated': result, 'real': judge['ood_pairs'][name]}
    print(f'4. Conditioning: all five words right in {conditioning["test"]["generated"]["all"]:.1%} of test, '
          f'{conditioning["ood"]["generated"]["all"]:.1%} of OOD')

    # 5. Quality: FID and KID against the real images of the same split, and between two groups of real images
    halves = torch.randperm(len(real['ood']), generator=torch.Generator().manual_seed(config.SEED))
    half = len(halves) // 2
    # every detail of the computation: the defaults of torchmetrics change between versions (hence the pin)
    quality = {'settings': {'library': f'torchmetrics {torchmetrics.__version__}',
                            'network': 'Inception-v3 of torch-fidelity, weights pt_inception-2015-12-05',
                            'features': 2048, 'input': 'floats in [0, 1] (normalize=True)',
                            'resize': '299x299 with torch.nn.functional.interpolate, bilinear, align_corners=False, '
                                      'antialias=True (the torchmetrics default, not the resize of torch-fidelity)',
                            'kid_subsets': config.KID_SUBSETS, 'kid_subset_size': config.KID_SUBSET_SIZE},
               'test': fid_kid(real['test'], from_pixels(generated['test']['images']), config),
               'ood': fid_kid(real['ood'], from_pixels(generated['ood']['images']), config),
               'reference_test': fid_kid(real['val'], real['test'], config),
               'reference_ood': fid_kid(real['ood'][halves[:half]], real['ood'][halves[half:]], config)}
    print(f'5. FID: test {quality["test"]["fid"]:.2f} (real {quality["reference_test"]["fid"]:.2f}), '
          f'OOD {quality["ood"]["fid"]:.2f} (real {quality["reference_ood"]["fid"]:.2f})')

    # 6. Diversity: the pairs of the images of each prompt, generated and real (all the splits, same caption)
    ImagePreprocessor(run_config).save_preview(
        [ImagePreprocessor.denormalize(image) for image in from_pixels(diverse['images'])], folder / 'diversity.png')
    captions = [caption for dataset in data.values() for caption in dataset.table['caption']]
    all_real = torch.cat(list(real.values()))                  # in the same order as captions
    ood_captions = set(data['ood'].table['caption'])
    n = config.DIVERSITY_IMAGES
    rows = []
    for k, prompt in enumerate(prompts):
        same = torch.tensor([caption == prompt for caption in captions])
        rows.append({'prompt': prompt, 'held_out': prompt in ood_captions,
                     'generated': mean_pair_distance(from_pixels(diverse['images'][k * n:(k + 1) * n])),
                     'real': mean_pair_distance(all_real[same]), 'real_images': int(same.sum())})
    diversity = {'images_per_prompt': n, 'distance': 'mean absolute pixel difference, pixels in [0, 1]',
                 'prompts': rows}
    for key, held_out in [('seen', False), ('held_out', True)]:
        group = [row for row in rows if row['held_out'] == held_out]
        diversity[key] = {name: sum(row[name] for row in group) / len(group) for name in ['generated', 'real']}
    print(f'6. Diversity: seen {diversity["seen"]["generated"]:.3f} (real {diversity["seen"]["real"]:.3f}), '
          f'OOD {diversity["held_out"]["generated"]:.3f} (real {diversity["held_out"]["real"]:.3f})')

    # 7. Parameters, sampling time and GPU memory
    parameters = {name: 0 if part is None else sum(p.numel() for p in part.parameters())
                  for name, part in [('total', model), ('text_encoder', model.text_encoder), ('unet', model.unet)]}
    batch_memory = [generated[split]['peak_memory_mb'] for split in ['test', 'ood']]
    sampling = {'device': device, 'batch_size': config.BATCH_SIZE,
                'seconds_per_image_in_batch': (generated['test']['seconds'] + generated['ood']['seconds'])
                                              / (len(data['test']) + len(data['ood'])),
                'peak_memory_mb_batch': None if None in batch_memory else max(batch_memory),
                'seconds_single_image': single_seconds, 'peak_memory_mb_single': single_memory}
    print(f'7. {parameters["total"]:,} parameters; {sampling["seconds_per_image_in_batch"]:.2f} s per image in a '
          f'group, {single_seconds:.1f} s alone')

    # 8. Everything in one file, for the report
    evaluation = {'experiment': run_dir.name, 'image_size': image_size,
                  'text_conditioning': run_config.TEXT_CONDITIONING, 'guidance': run_config.GUIDANCE_SCALE,
                  'parameters': parameters, 'sampling': sampling, 'conditioning': conditioning, 'quality': quality,
                  'diversity': diversity}
    with open(folder / 'evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f'Done: {folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained experiment (settings in config.py).')
    parser.add_argument('run_dir', type=Path, help='folder of the experiment, e.g. runs/conditional_32')
    main(parser.parse_args().run_dir)
