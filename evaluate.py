"""Evaluation of a trained experiment on the captions of the ordinary test and of the held-out combinations (OOD).

Run it with:  python evaluate.py runs/conditional_32     (after train.py and train_classifier.py)
It generates one image per row of the test and OOD splits and measures the conditioning (with the attribute
classifier), the quality (FID and KID, with real-vs-real references), the diversity across seeds (SAMPLE_PROMPTS),
the parameters, the sampling time and the GPU memory. Everything goes to runs/<experiment>/evaluation/.
"""

import json
import time

import torch

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
