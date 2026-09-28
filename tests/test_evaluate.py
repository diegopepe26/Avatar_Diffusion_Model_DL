"""Tests of the evaluation of an experiment. Run them with:  python -m pytest tests/"""

import json

import pytest
import torch

from config import Config
from evaluate import check_run, generate_images, generate_or_load, load_judge
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
