"""Tests of the generation step, the model, the sampling loop and the training functions.

Run them with:  python -m pytest tests/
"""

import torch

from config import Config
from models.noise_scheduler import NoiseScheduler


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
