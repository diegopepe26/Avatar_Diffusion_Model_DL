"""Tests of the generation step, the model, the sampling loop and the training functions.

Run them with:  python -m pytest tests/
"""

import torch

from config import Config
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from models.sampling import sample
from preprocessing.vocabulary import Vocabulary

CAPTIONS = [
    'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard',
    'a cartoon avatar with dark skin, short black hair, sunglasses and no beard',
]


def make_model(text_conditioning=True, num_timesteps=1000):
    """A model, its config and caption ids, with a small vocabulary built from CAPTIONS (no data/ needed)."""
    config = Config()
    config.TEXT_CONDITIONING = text_conditioning
    config.NUM_TIMESTEPS = num_timesteps
    vocabulary = Vocabulary(config)
    vocabulary.build(CAPTIONS)
    tokens = torch.tensor([vocabulary.encode(caption) for caption in CAPTIONS])
    return DiffusionModel(config, vocabulary), config, vocabulary, tokens


def test_step_at_zero_with_the_real_noise_gives_the_clean_image():
    scheduler = NoiseScheduler(Config())
    clean = torch.rand(2, 3, 32, 32) * 2 - 1
    noise = torch.randn(2, 3, 32, 32)
    t = torch.zeros(2, dtype=torch.long)
    # if the model guessed the noise exactly, the last step gives back the clean image
    assert torch.allclose(scheduler.step(scheduler.add_noise(clean, t, noise), 0, noise), clean, atol=1e-5)


def test_step_is_finite_at_the_first_step():
    scheduler = NoiseScheduler(Config())
    x_t = torch.randn(2, 3, 32, 32)
    # t = 999: sqrt(alpha_bar) ~ 0.00005, the clean image estimate must stay under control
    assert torch.isfinite(scheduler.step(x_t, 999, torch.randn(2, 3, 32, 32))).all()


def test_step_depends_on_the_seed():
    scheduler = NoiseScheduler(Config())
    x_t = torch.randn(2, 3, 32, 32)
    noise = torch.randn(2, 3, 32, 32)
    first = scheduler.step(x_t, 500, noise, torch.Generator().manual_seed(0))
    same = scheduler.step(x_t, 500, noise, torch.Generator().manual_seed(0))
    other = scheduler.step(x_t, 500, noise, torch.Generator().manual_seed(1))
    assert torch.equal(first, same)
    assert not torch.equal(first, other)


def test_diffusion_model():
    model, _, vocabulary, tokens = make_model()
    x_t = torch.randn(2, 3, 32, 32)
    t = torch.tensor([10, 900])
    assert model(x_t, t, tokens).shape == (2, 3, 32, 32)
    assert model.empty_tokens.tolist() == vocabulary.encode('')     # <bos>, <eos>, then <pad>
    unconditional, _, _, _ = make_model(text_conditioning=False)
    assert unconditional.text_encoder is None
    assert unconditional(x_t, t, tokens).shape == (2, 3, 32, 32)


def test_sample():
    for text_conditioning in (True, False):
        model, config, _, tokens = make_model(text_conditioning, num_timesteps=10)   # 10 steps: fast
        model.eval()
        scheduler = NoiseScheduler(config)
        images = sample(model, scheduler, tokens, config, seed=0)
        assert images.shape == (2, 3, 32, 32)
        assert torch.isfinite(images).all()
        assert torch.equal(images, sample(model, scheduler, tokens, config, seed=0))       # same seed, same images
        assert not torch.equal(images, sample(model, scheduler, tokens, config, seed=1))
