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


def test_add_noise_extremes():
    scheduler = NoiseScheduler(Config())
    images = torch.rand(4, 3, 32, 32) * 2 - 1      # values in [-1, 1], like CartoonDataset
    noise = torch.randn(4, 3, 32, 32)
    first = scheduler.add_noise(images, torch.zeros(4, dtype=torch.long), noise)
    last = scheduler.add_noise(images, torch.full((4,), 999), noise)
    assert torch.allclose(first, images, atol=0.05)   # t = 0: almost the clean image
    assert torch.allclose(last, noise, atol=0.05)     # t = 999: almost only noise


def test_add_noise_one_step_per_image():
    scheduler = NoiseScheduler(Config())
    images = torch.rand(3, 3, 64, 64) * 2 - 1      # 64x64: IMAGE_SIZE can also be 64
    noise = torch.randn(3, 3, 64, 64)
    t = torch.tensor([0, 500, 999])
    noisy = scheduler.add_noise(images, t, noise)
    assert noisy.shape == images.shape
    for i in range(3):   # every image with the alpha_bar of its own step
        alpha_bar = scheduler.alpha_bars[t[i]]
        expected = alpha_bar.sqrt() * images[i] + (1 - alpha_bar).sqrt() * noise[i]
        assert torch.allclose(noisy[i], expected)


def test_add_noise_step_out_of_range():
    scheduler = NoiseScheduler(Config())
    images = torch.zeros(1, 3, 32, 32)
    with pytest.raises(IndexError):   # t goes from 0 to 999: 1000 does not exist
        scheduler.add_noise(images, torch.tensor([1000]), torch.randn(1, 3, 32, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_add_noise_on_gpu():
    scheduler = NoiseScheduler(Config())            # the table stays on the CPU
    images = torch.rand(2, 3, 32, 32, device='cuda') * 2 - 1
    noise = torch.randn(2, 3, 32, 32, device='cuda')
    t_gpu = torch.randint(0, 1000, (2,), device='cuda')
    assert scheduler.add_noise(images, t_gpu, noise).device.type == 'cuda'
    t_cpu = torch.randint(0, 1000, (2,))            # t made without device=...: must work too
    assert scheduler.add_noise(images, t_cpu, noise).device.type == 'cuda'
