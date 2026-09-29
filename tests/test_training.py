"""Tests of the generation step, the model, the sampling loop and the training functions.

Run them with:  python -m pytest tests/
"""

import copy

import pytest
import torch
from PIL import Image

from config import Config
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from models.sampling import sample
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.vocabulary import Vocabulary
from train import diffusion_loss, diffusion_noise, save_checkpoint, update_ema

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


def test_save_preview(tmp_path):
    config = Config()
    images = [ImagePreprocessor.denormalize(torch.rand(3, 32, 32) * 2 - 1) for _ in range(16)]
    ImagePreprocessor(config).save_preview(images, tmp_path / 'grid.png')
    assert Image.open(tmp_path / 'grid.png').size == (8 * 128, 2 * 128)    # 8 columns, 2 rows, 4x enlarged


def test_diffusion_loss():
    model, config, _, tokens = make_model()
    model.eval()                                                          # no dropout, as in the validation
    scheduler = NoiseScheduler(config)
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = diffusion_loss(model, scheduler, images, tokens, torch.Generator().manual_seed(0))
    assert loss.dim() == 0                                                # one number
    same = diffusion_loss(model, scheduler, images, tokens, torch.Generator().manual_seed(0))
    assert torch.equal(loss, same)                                        # same seed, same t and noise
    loss.backward()
    assert model.text_encoder.embedding.weight.grad is not None           # the text encoder learns too


def test_offset_noise_shifts_each_channel_by_one_number():
    plain = diffusion_noise((2, 3, 8, 8), 0.0, torch.Generator().manual_seed(0))
    # OFFSET_NOISE = 0 is the plain noise of DDPM, with the same random numbers as before (the 32x32 runs)
    assert torch.equal(plain, torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)))
    shifted = diffusion_noise((2, 3, 8, 8), 0.1, torch.Generator().manual_seed(0))
    offset = shifted - plain                                              # the same numbers first, then the offset
    # one number for all the pixels of a channel of an image, different between channels
    assert torch.allclose(offset, offset[:, :, :1, :1].expand_as(offset))
    assert len(torch.unique(offset[:, :, 0, 0])) == 6                     # 2 images x 3 channels


def test_update_ema():
    model, config, _, _ = make_model()
    ema = copy.deepcopy(model)
    with torch.no_grad():
        for weight in model.parameters():
            weight.add_(1.0)                                   # the model moves away from the EMA
    old = next(ema.parameters()).clone()
    new = next(model.parameters()).clone()
    update_ema(ema, model, step=0, config=config)              # first step: decay = min(0.999, 1/10) = 0.1
    assert torch.allclose(next(ema.parameters()), 0.1 * old + 0.9 * new)
    old = next(ema.parameters()).clone()
    update_ema(ema, model, step=10_000, config=config)         # later: decay = EMA_DECAY = 0.999
    assert torch.allclose(next(ema.parameters()), 0.999 * old + 0.001 * new)


def test_interrupted_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'last.pt'
    save_checkpoint({'epoch': 1}, path)

    def interrupted_save(checkpoint, file):
        open(file, 'wb').write(b'half a checkpoint')      # the write stops halfway (e.g. Colab disconnects)
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint({'epoch': 2}, path)
    monkeypatch.undo()
    assert torch.load(path)['epoch'] == 1                  # the previous checkpoint is still there and readable
