"""Forward diffusion: the fixed recipe that brings a clean image to the noise level of step t."""

import math

import torch


class NoiseScheduler:
    """Cosine noise schedule (Nichol & Dhariwal 2021, Improved DDPM) and the noising of the images.

    Nothing to train: the same step t always gives the same amount of noise.
    """

    def __init__(self, config):
        """Compute, once, the noise of every step and how much of the image is left after it.

        Args:
            config: the project Config (uses NUM_TIMESTEPS).
        """
        self.num_timesteps = config.NUM_TIMESTEPS
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float64) / self.num_timesteps   # 0, 1/T, ..., 1
        # cosine curve of the paper; the 0.008 offset keeps the first steps from being too small
        curve = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        # noise added by the single step t, at most 0.999: the last step would otherwise be 1
        # (image completely gone) and the generation would divide by zero
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        self.betas = betas.float()                                    # (T,)
        # share of the image left after step t: product of (1 - beta) from step 0 to t
        # (computed in float64: it multiplies 1000 numbers)
        self.alpha_bars = torch.cumprod(1 - betas, dim=0).float()     # (T,)

    def add_noise(self, images, t, noise):
        """Bring every image to the noise level of its step t, in one go.

        The t small noise additions of the chain add up to a single one:
        x_t = sqrt(alpha_bar_t) * image + sqrt(1 - alpha_bar_t) * noise.

        Args:
            images: (B, 3, H, W) clean images in [-1, 1].
            t: (B,) integer steps between 0 and NUM_TIMESTEPS - 1, one per image.
            noise: (B, 3, H, W) Gaussian noise, passed from outside: the training needs it for the loss.

        Returns:
            (B, 3, H, W) noisy images x_t, on the device of images.
        """
        # the table lives on the CPU: move it to the device of the images, take one value per image
        # and shape it (B, 1, 1, 1), so it applies to all the channels and pixels of that image
        alpha_bar = self.alpha_bars.to(images.device)[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise
