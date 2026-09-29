"""Tests of the evaluation of an experiment. Run them with:  python -m pytest tests/"""

import json

import pytest
import torch

import evaluate
from config import Config
from evaluate import check_run, disagreement_line, generate_images, generate_or_load, load_judge, mean_pair_distance
from models.attribute_classifier import AttributeClassifier
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


def test_check_run_stops_before_the_generation(tmp_path):
    config = Config()
    with pytest.raises(ValueError):
        check_run(tmp_path, config)                           # no last.pt: never trained
    other = 64 if config.IMAGE_SIZE == 32 else 32
    torch.save({'settings': {'IMAGE_SIZE': other}}, tmp_path / 'last.pt')
    with pytest.raises(ValueError):
        check_run(tmp_path, config)                           # the real images in data/ have another size
    torch.save({'settings': {'IMAGE_SIZE': config.IMAGE_SIZE}}, tmp_path / 'last.pt')
    assert check_run(tmp_path, config) == config.IMAGE_SIZE


def test_load_judge_checks_that_accuracy_json_is_of_the_same_training(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'RUNS_DIR', tmp_path)            # instead of runs/
    config = Config()
    with pytest.raises(ValueError):
        load_judge(config, 32)                                   # no classifier trained at this size
    folder = tmp_path / 'classifier_32'
    folder.mkdir()
    torch.save({'weights': AttributeClassifier(config).state_dict(),
                'settings': {'IMAGE_SIZE': 32, 'CLASSIFIER_CHANNELS': config.CLASSIFIER_CHANNELS},
                'epoch': 40, 'val_accuracy': 0.98}, folder / 'classifier.pt')
    with pytest.raises(ValueError):
        load_judge(config, 32)                                   # accuracy.json missing: training interrupted
    (folder / 'accuracy.json').write_text(json.dumps({'epoch': 30}))
    with pytest.raises(ValueError):
        load_judge(config, 32)                                   # accuracy.json of another training
    (folder / 'accuracy.json').write_text(json.dumps({'epoch': 40}))
    classifier, report = load_judge(config, 32)
    assert report['epoch'] == 40 and not classifier.training


def test_diversity_is_the_mean_over_all_the_pairs():
    black, white = -torch.ones(3, 32, 32), torch.ones(3, 32, 32)
    # three pairs: black-black 0, black-white 1, black-white 1
    assert mean_pair_distance(torch.stack([black, black, white])) == pytest.approx(2 / 3)


def test_disagreement_line_lists_only_the_wrong_words():
    asked = {'face_color': 'dark', 'hair_color': 'black', 'hair': 'long', 'glasses': 'glasses', 'facial_hair': 'a beard'}
    seen = dict(asked, hair='medium', glasses='no glasses')
    assert disagreement_line(5, 'a cartoon avatar', asked, seen) == \
        '5. a cartoon avatar — hair: asked long, sees medium; glasses: asked glasses, sees no glasses'


def test_fid_and_kid_get_the_exact_pixels(monkeypatch):
    received = []       # (normalize, images) of every update

    class Recorder:
        """Stands in for FID and KID of torchmetrics (no Inception needed): keeps what fid_kid gives them."""

        def __init__(self, **settings):
            self.kid, self.normalize = 'subsets' in settings, settings['normalize']

        def to(self, device):
            return self

        def update(self, images, real):
            received.append((self.normalize, images.cpu()))

        def compute(self):
            return (torch.tensor(0.0), torch.tensor(0.0)) if self.kid else torch.tensor(0.0)

    monkeypatch.setattr(evaluate, 'FrechetInceptionDistance', Recorder)
    monkeypatch.setattr(evaluate, 'KernelInceptionDistance', Recorder)
    pixels = torch.arange(256, dtype=torch.uint8).repeat(1, 3, 1, 1)     # every value 0-255 once: (1, 3, 1, 256)
    evaluate.fid_kid(pixels, pixels, Config())
    # the integers themselves, not floats that torchmetrics would turn back into integers (a few one step lower)
    assert len(received) == 4 and all(not normalize and torch.equal(images, pixels) for normalize, images in received)
