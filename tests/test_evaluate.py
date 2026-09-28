"""Tests of the evaluation of an experiment. Run them with:  python -m pytest tests/"""

import torch

from config import Config
from evaluate import generate_images, generate_or_load
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from preprocessing.vocabulary import Vocabulary

CAPTION = 'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard'


def small_model(copies):
    """An untrained model with 10 noise steps (fast), groups of 2 images, and the ids of copies of CAPTION."""
    config = Config()
    config.NUM_TIMESTEPS = 10
    config.BATCH_SIZE = 2
    vocabulary = Vocabulary(config)
    vocabulary.build([CAPTION])
    tokens = torch.tensor([vocabulary.encode(CAPTION)] * copies)
    return DiffusionModel(config, vocabulary).eval(), NoiseScheduler(config), config, tokens


def test_every_group_of_images_has_its_own_seed():
    model, scheduler, config, tokens = small_model(copies=4)       # two groups of 2, same caption
    images = generate_images(model, scheduler, config, tokens, first_seed=0)
    assert images.shape == (4, 3, 32, 32) and images.dtype == torch.uint8
    # the first image of each group: with the same seed for both groups they would be equal
    assert not torch.equal(images[0], images[2])
    assert torch.equal(images, generate_images(model, scheduler, config, tokens, first_seed=0))   # reproducible


def test_saved_images_are_read_not_generated_again(tmp_path):
    saved = {'images': torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8), 'seconds': 12.5,
             'peak_memory_mb': None}
    torch.save(saved, tmp_path / 'generated_test.pt')
    # no model and no captions: generating again would fail
    loaded = generate_or_load(tmp_path / 'generated_test.pt', None, None, Config(), None, first_seed=0)
    assert torch.equal(loaded['images'], saved['images']) and loaded['seconds'] == 12.5
