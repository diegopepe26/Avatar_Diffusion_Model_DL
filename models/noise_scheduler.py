"""Forward diffusion, the fixed recipe that brings a clean image to the noise level of step t."""

import math

import torch


class NoiseScheduler:
    """Cosine noise schedule and the noising of the images.

    There are T = NUM_TIMESTEPS noise steps (1000), from t = 0 (almost the clean image) to t = T - 1 (pure noise).
    There is nothing to train, the same step t always gives the same amount of noise.
    """

    def __init__(self, config):
        """Compute once the noise of every step and how much of the image is left after it.

        Args:
            config: the project Config (uses NUM_TIMESTEPS).
        """
        self.num_timesteps = config.NUM_TIMESTEPS
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float64) / self.num_timesteps   # 0, 1/T, ..., 1

        # cosine curve, where the 0.008 offset prevents the first steps from being too small
        curve = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2

        # noise added by the single step t, from the ratio of two consecutive points of the curve. The curve ends
        # at exactly 0, so the last beta would be 1, the last alpha_bar would be 0 and step() would divide by
        # sqrt(0) when it estimates the clean image. With at most 0.999 the last alpha_bar stays above 0
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)

        # betas and alpha_bars are computed in float64, more precise on the tiny values near the end and on the
        # product of 1000 numbers, then kept in float32 like the images and the weights of the model.
        self.betas = betas.float()                                    # (T,)

        # share of the image left after step t, the product of (1 - beta) from step 0 to t
        self.alpha_bars = torch.cumprod(1 - betas, dim=0).float()     # (T,)

    def add_noise(self, images, t, noise):
        """Bring every image to the noise level of its step t, in one go.

        The t small noise additions of the chain add up to a single one,
        x_t = sqrt(alpha_bar_t) * image + sqrt(1 - alpha_bar_t) * noise.

        Args:
            images: (B, 3, H, W) clean images in [-1, 1].
            t: (B,) integer steps between 0 and NUM_TIMESTEPS - 1, one per image.
            noise: (B, 3, H, W) Gaussian noise, passed from outside because the training needs it for the loss.

        Returns:
            (B, 3, H, W) noisy images x_t, on the device of images.
        """
        # the table lives on the CPU, so it is moved to the device of the images. One value per image, shaped
        # (B, 1, 1, 1) so that it applies to all the channels and pixels of that image
        alpha_bar = self.alpha_bars.to(images.device)[t].view(-1, 1, 1, 1)

        return alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise

    def step(self, x_t, t, predicted_noise, generator=None):
        """One step back of the generation, from the images at step t to the images at step t - 1, with less noise.

        First it estimates the clean image x_0 from x_t and the predicted noise (add_noise done in reverse), and keeps
        its values in [-1, 1]. Then it mixes x_0 and x_t, with weights that change with t, and adds some new random
        noise. The new noise gets smaller as t goes down. At t = 0 it returns x_0, with no new noise.

        Args:
            x_t: (B, 3, H, W) images at step t.
            t: integer step, the same for the whole batch (from NUM_TIMESTEPS - 1 down to 0).
            predicted_noise: (B, 3, H, W) the noise predicted by the model for x_t.
            generator: torch.Generator of the fresh noise, or None.

        Returns:
            (B, 3, H, W) images at step t - 1, on the device of x_t.
        """
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]

        # 1. the clean image according to the model, by reversing add_noise
        x_0 = (x_t - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

        # 2. real images are in [-1, 1]; near t = 999, dividing by sqrt(alpha_bar), about 0.00005, blows up any error
        x_0 = x_0.clamp(-1, 1)
        if t == 0:
            return x_0   # at the last step the clean image

        # 3. a step towards x_0, the mean of q(x_(t-1) | x_t, x_0)
        alpha_bar_previous = self.alpha_bars[t - 1]
        mean = (alpha_bar_previous.sqrt() * beta / (1 - alpha_bar) * x_0
                + (1 - beta).sqrt() * (1 - alpha_bar_previous) / (1 - alpha_bar) * x_t)

        # 4. the fresh noise keeps the images varied and lets the next steps correct the errors
        variance = beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)

        return mean + variance.sqrt() * noise
