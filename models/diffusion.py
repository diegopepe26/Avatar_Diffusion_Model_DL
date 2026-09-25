"""The part of the diffusion model that learns: text encoder + UNet, from noisy images to predicted noise."""

import torch
from torch import nn

from models.text_encoder import TextEncoder
from models.unet import UNet


class DiffusionModel(nn.Module):
    """Text encoder and UNet together: one object to train, copy for the EMA and save.

    The noise scheduler is not here: it has no weights, it is the fixed recipe of the noise.
    """

    def __init__(self, config, vocabulary):
        """Args:
            config: the project Config (uses TEXT_CONDITIONING and the settings of the two networks).
            vocabulary: the loaded Vocabulary (for the text encoder and the empty caption).
        """
        super().__init__()
        # no text encoder in the unconditional baseline
        self.text_encoder = TextEncoder(config, vocabulary) if config.TEXT_CONDITIONING else None
        self.unet = UNet(config)
        # the empty caption '' (<bos>, <eos>, then <pad>): the "no text" of classifier-free guidance.
        # Not only <pad>: that would hide every word and the attention would give NaN.
        # A buffer: it follows the model to the GPU and into the checkpoint.
        self.register_buffer('empty_tokens', torch.tensor(vocabulary.encode('')))

    def forward(self, x_t, t, tokens):
        """Predict the noise of x_t.

        Args:
            x_t: (B, 3, H, W) noisy images.
            t: (B,) integer steps.
            tokens: (B, L) caption ids from Vocabulary.encode; ignored without text encoder.

        Returns:
            (B, 3, H, W) predicted noise.
        """
        if self.text_encoder is None:
            return self.unet(x_t, t)
        context, pad_mask = self.text_encoder(tokens)
        return self.unet(x_t, t, context, pad_mask)
