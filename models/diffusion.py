"""The part of the diffusion model that learns, made of the text encoder and the UNet."""

import torch
from torch import nn

from models.text_encoder import TextEncoder
from models.unet import UNet


class DiffusionModel(nn.Module):
    """Text encoder followed by the UNet. It predicts the noise of the noisy images, given their step t and their
    caption."""

    def __init__(self, config, vocabulary):
        """Args:
            config: the project Config (uses TEXT_CONDITIONING and the settings of the two networks).
            vocabulary: the loaded Vocabulary (for the text encoder and the empty caption).
        """
        super().__init__()

        # no text encoder in the unconditional baseline
        self.text_encoder = TextEncoder(config, vocabulary) if config.TEXT_CONDITIONING else None
        self.unet = UNet(config)

        # the empty caption '' (<bos>, <eos>, then <pad>) is the "no text" of classifier-free guidance.
        # <bos> and <eos> stay because with only <pad> every token would be hidden and the softmax of the
        # attention would give NaN.
        # register_buffer and not a plain attribute, so model.to(device) also moves these ids to the GPU and the
        # checkpoint saves them with the weights
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
