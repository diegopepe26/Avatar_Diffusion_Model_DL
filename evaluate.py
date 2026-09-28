"""Evaluation of a trained experiment on the captions of the ordinary test and of the held-out combinations (OOD).

Run it with:  python evaluate.py runs/conditional_32     (after train.py and train_classifier.py)
It generates one image per row of the test and OOD splits and measures the conditioning (with the attribute
classifier), the quality (FID and KID, with real-vs-real references), the diversity across seeds (SAMPLE_PROMPTS),
the parameters, the sampling time and the GPU memory. Everything goes to runs/<experiment>/evaluation/.
"""

import json
import time

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from models.attribute_classifier import load_classifier
from models.sampling import sample


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
