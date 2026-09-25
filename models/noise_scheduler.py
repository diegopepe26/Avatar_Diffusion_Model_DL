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

    def step(self, x_t, t, predicted_noise, generator=None):
        """One step back of the generation: from the images at step t to the images at step t - 1.

        DDPM reverse step: estimate the clean image by turning add_noise around, keep it in [-1, 1]
        (as the official DDPM code does), move towards it with the mean of q(x_(t-1) | x_t, x_0) and
        add fresh noise with variance beta_tilde ("fixed small" of DDPM). No fresh noise at t = 0.

        Args:
            x_t: (B, 3, H, W) images at step t.
            t: integer step, the same for the whole batch (from NUM_TIMESTEPS - 1 down to 0).
            predicted_noise: (B, 3, H, W) the noise predicted by the model for x_t.
            generator: torch.Generator of the fresh noise (same seed, same images), or None.

        Returns:
            (B, 3, H, W) images at step t - 1, on the device of x_t.
        """
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        # 1. the clean image according to the model: add_noise turned around
        x_0 = (x_t - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        # 2. real images are in [-1, 1]; near t = 999, dividing by sqrt(alpha_bar) ~ 0.00005 blows up any error
        x_0 = x_0.clamp(-1, 1)
        if t == 0:
            return x_0   # last step: the clean image (here the mean below is exactly x_0)
        # 3. a step towards x_0: mean of q(x_(t-1) | x_t, x_0), equation 7 of DDPM
        alpha_bar_previous = self.alpha_bars[t - 1]
        mean = (alpha_bar_previous.sqrt() * beta / (1 - alpha_bar) * x_0
                + (1 - beta).sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar) * x_t)
        # 4. fresh noise: keeps the images varied and lets the next steps correct the errors
        variance = beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)
        return mean + variance.sqrt() * noise
