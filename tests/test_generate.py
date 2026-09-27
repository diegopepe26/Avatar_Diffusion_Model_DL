"""Tests of the generation functions. Run them with:  python -m pytest tests/"""

import torch

from config import Config
from generate import generate, is_held_out, load_model
from models.diffusion import DiffusionModel
from preprocessing.vocabulary import Vocabulary

CAPTION = 'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard'


def save_small_run(tmp_path, monkeypatch, image_size=32):
    """An untrained model saved like train.py does, with 10 noise steps (fast), and its vocabulary."""
    config = Config()
    config.IMAGE_SIZE = image_size
    config.NUM_TIMESTEPS = 10
    vocabulary = Vocabulary(config)
    vocabulary.build([CAPTION])
    vocabulary.save(tmp_path / 'vocabulary.json')
    monkeypatch.setattr(Config, 'VOCABULARY_FILE', tmp_path / 'vocabulary.json')   # instead of data/
    settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    torch.save({'ema': DiffusionModel(config, vocabulary).state_dict(), 'settings': settings}, run_dir / 'last.pt')
    return run_dir


def test_one_seed_per_image(tmp_path, monkeypatch):
    model, scheduler, config, vocabulary = load_model(save_small_run(tmp_path, monkeypatch))
    together = generate(model, scheduler, config, vocabulary, CAPTION, [7, 8], guidance=3.0)
    alone = generate(model, scheduler, config, vocabulary, CAPTION, [8], guidance=3.0)
    assert list(together[1].getdata()) == list(alone[0].getdata())    # seed 8: the same image, alone or in a group
    assert list(together[0].getdata()) != list(together[1].getdata())


def test_settings_come_from_the_checkpoint(tmp_path, monkeypatch):
    run_dir = save_small_run(tmp_path, monkeypatch, image_size=64)    # while config.py says 32
    model, scheduler, config, vocabulary = load_model(run_dir)
    assert config.IMAGE_SIZE == 64 and model.unet.outer_level
    image, = generate(model, scheduler, config, vocabulary, CAPTION, [0], guidance=3.0)
    assert image.size == (64, 64)


def test_is_held_out():
    config = Config()
    words = {'face_color': 'dark', 'hair': 'long', 'hair_color': 'black', 'glasses': 'no glasses',
             'facial_hair': 'no beard'}
    assert is_held_out(words, config)                   # dark skin + long hair
    words['face_color'] = 'tan'
    assert not is_held_out(words, config)               # seen in training
