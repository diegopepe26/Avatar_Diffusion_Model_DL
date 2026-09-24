"""Tests of the noise scheduler. Run them with:  python -m pytest tests/"""

import pytest
import torch

from config import Config
from models.noise_scheduler import NoiseScheduler


def test_cosine_curve():
    scheduler = NoiseScheduler(Config())
    assert scheduler.betas.shape == (1000,)
    assert scheduler.alpha_bars.shape == (1000,)
    assert scheduler.alpha_bars[0] > 0.999                                   # step 0: almost the clean image
    assert torch.all(scheduler.alpha_bars[1:] < scheduler.alpha_bars[:-1])   # always less image left
    assert abs(scheduler.alpha_bars[499] - 0.49) < 0.01                      # cosine: about half at the middle
    assert scheduler.alpha_bars[999] < 0.001                                 # last step: pure noise
    assert torch.all(scheduler.betas > 0) and torch.all(scheduler.betas <= 0.999)
