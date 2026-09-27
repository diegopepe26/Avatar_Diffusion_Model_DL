"""UNet denoiser: predicts the noise added to x_t, given the step t and, optionally, the caption."""

import torch
from torch import nn
from torch.nn import functional as F

from models.attention import build_attention
from models.sinusoidal import sinusoidal_embedding


class ResBlock(nn.Module):
    """Residual block of DDPM: two 3x3 convolutions, with the time embedding added in between."""

    def __init__(self, in_channels, out_channels, config):
        """Args:
            in_channels: channels of the input.
            out_channels: channels of the output.
            config: the project Config (uses TIME_DIM, DROPOUT).
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)      # 32 groups, as in DDPM
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # brings the time embedding to one value per channel
        self.time_projection = nn.Linear(config.TIME_DIM, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # the input is added to the output: a 1x1 convolution adapts it when the channels change
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time):
        """Args:
            x: (B, in_channels, H, W).
            time: (B, TIME_DIM), the time embedding of the step t of every image.

        Returns:
            (B, out_channels, H, W).
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # the same value on every pixel of a channel: [:, :, None, None] makes it (B, C, 1, 1)
        h = h + self.time_projection(F.silu(time))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.shortcut(x) + h


class CrossAttention(nn.Module):
    """The pixels read the words of the caption: queries from the image, keys and values from the text.

    The attention itself is MultiHeadAttention of attention.py (or nn.MultiheadAttention, see
    Config.ATTENTION): this block only turns the image into a sequence of tokens and back.
    """

    def __init__(self, channels, config):
        """Args:
            channels: channels of the image at this level (the size of the attention).
            config: the project Config (uses D_MODEL, the size of the text vectors, and the attention settings).
        """
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        # W_K and W_V take the words (D_MODEL numbers) straight to the channels of the level
        self.attention = build_attention(config, channels, kdim=config.D_MODEL)

    def forward(self, x, context, pad_mask):
        """Args:
            x: (B, C, H, W) image features.
            context: (B, L, D_MODEL) the words, from the TextEncoder.
            pad_mask: (B, L) bool, True where the token is <pad>.

        Returns:
            (B, C, H, W): x plus what the pixels read from the text.
        """
        batch, channels, height, width = x.shape
        # every pixel becomes a token: (B, C, H, W) -> (B, H*W, C)
        pixels = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attention(pixels, context, context, key_padding_mask=pad_mask)
        # back to an image, added to the input (residual)
        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)


class UNetBlock(nn.Module):
    """A ResBlock, followed by a CrossAttention where the level has one and the text is used."""

    def __init__(self, in_channels, out_channels, config, attention):
        """Args:
            in_channels: channels of the input.
            out_channels: channels of the output.
            config: the project Config (uses TEXT_CONDITIONING and the block settings).
            attention: True on the levels with cross-attention (16x16 and 8x8).
        """
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, config)
        # None: no cross-attention on this level, or unconditional model (TEXT_CONDITIONING = False)
        self.attention = CrossAttention(out_channels, config) if attention and config.TEXT_CONDITIONING else None

    def forward(self, x, time, context, pad_mask):
        """Args:
            x: (B, in_channels, H, W).
            time: (B, TIME_DIM).
            context: (B, L, D_MODEL), or None without text.
            pad_mask: (B, L), or None without text.

        Returns:
            (B, out_channels, H, W).
        """
        x = self.res(x, time)
        if self.attention is not None:
            x = self.attention(x, context, pad_mask)
        return x


class UNet(nn.Module):
    """Two blocks per level, skip connections by concatenation, the bottom always at 8x8.

    IMAGE_SIZE = 32: three resolutions (32, 16, 8). IMAGE_SIZE = 64: the same network with one more level
    outside, at 64x64 (64, 32, 16, 8).
    """

    def __init__(self, config):
        """Args:
            config: the project Config (uses UNET_CHANNELS, TIME_DIM, TEXT_CONDITIONING and the block settings).
        """
        super().__init__()
        if config.IMAGE_SIZE not in (32, 64):
            raise ValueError(f'IMAGE_SIZE must be 32 or 64, not {config.IMAGE_SIZE}')
        c1, c2, c3 = config.UNET_CHANNELS          # 64, 128, 256: channels at 32x32, 16x16, 8x8
        self.base_channels = c1
        self.text_conditioning = config.TEXT_CONDITIONING
        # 64x64: one more level outside the 32x32 network (64 -> 32), so the bottom is still 8x8, the
        # cross-attention still at 16x16 and 8x8 and every bottom pixel still sees the whole image.
        # At 32x32 its layers do not exist: the network and its checkpoints stay as they were.
        self.outer_level = config.IMAGE_SIZE == 64
        self.image_size = config.IMAGE_SIZE
        # time embedding: sin/cos of t (the same formula as the positions of the text), then a small MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(c1, config.TIME_DIM),
            nn.SiLU(),
            nn.Linear(config.TIME_DIM, config.TIME_DIM),
        )
        self.conv_in = nn.Conv2d(3, c1, 3, padding=1)
        if self.outer_level:
            self.down0 = nn.ModuleList([UNetBlock(c1, c1, config, False), UNetBlock(c1, c1, config, False)])
            self.downsample0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                      # 64 -> 32
        # down: two blocks per level, then a stride-2 convolution halves the size
        self.down1 = nn.ModuleList([UNetBlock(c1, c1, config, False), UNetBlock(c1, c1, config, False)])
        self.downsample1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                          # 32 -> 16
        self.down2 = nn.ModuleList([UNetBlock(c1, c2, config, True), UNetBlock(c2, c2, config, True)])
        self.downsample2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)                          # 16 -> 8
        # bottom: ResBlock -> CrossAttention -> ResBlock, as in the papers
        self.middle = nn.ModuleList([UNetBlock(c2, c3, config, True), UNetBlock(c3, c3, config, False)])
        # up: nearest interpolation + 3x3 convolution (no checkerboard artifacts), then the skip is concatenated
        self.upsample2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c3, c3, 3, padding=1))
        self.up2 = nn.ModuleList([UNetBlock(c3 + c2, c2, config, True), UNetBlock(c2, c2, config, True)])
        self.upsample1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c2, c2, 3, padding=1))
        self.up1 = nn.ModuleList([UNetBlock(c2 + c1, c1, config, False), UNetBlock(c1, c1, config, False)])
        if self.outer_level:
            self.upsample0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c1, c1, 3, padding=1))
            self.up0 = nn.ModuleList([UNetBlock(c1 + c1, c1, config, False), UNetBlock(c1, c1, config, False)])
        self.conv_out = nn.Sequential(nn.GroupNorm(32, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1))

    def forward(self, x_t, t, context=None, pad_mask=None):
        """Predict the noise that NoiseScheduler.add_noise put into x_t.

        Args:
            x_t: (B, 3, IMAGE_SIZE, IMAGE_SIZE) noisy images.
            t: (B,) integer steps of the images.
            context: (B, L, D_MODEL) from the TextEncoder; not needed if TEXT_CONDITIONING is False.
            pad_mask: (B, L) bool from the TextEncoder, True where the token is <pad>.

        Returns:
            (B, 3, H, W) predicted noise.
        """
        if self.text_conditioning and context is None:
            raise ValueError('TEXT_CONDITIONING is True: the UNet needs the context of the TextEncoder')
        if x_t.shape[-1] != self.image_size:
            # otherwise the network runs anyway, with the wrong number of levels (e.g. IMAGE_SIZE changed on a
            # Config object: the images folder images_32 was fixed when config.py was read)
            raise ValueError(f'the images are {x_t.shape[-1]}x{x_t.shape[-1]} but IMAGE_SIZE is {self.image_size}: '
                             f'change IMAGE_SIZE in config.py and run prepare_data.py')
        time = self.time_mlp(sinusoidal_embedding(t, self.base_channels))    # (B, TIME_DIM)
        x = self.conv_in(x_t)
        if self.outer_level:
            for block in self.down0:
                x = block(x, time, context, pad_mask)
            skip0 = x                                                # (B, c1, 64, 64)
            x = self.downsample0(x)
        for block in self.down1:
            x = block(x, time, context, pad_mask)
        skip1 = x                                                    # (B, c1, 32, 32)
        x = self.downsample1(x)
        for block in self.down2:
            x = block(x, time, context, pad_mask)
        skip2 = x                                                    # (B, c2, 16, 16)
        x = self.downsample2(x)
        for block in self.middle:
            x = block(x, time, context, pad_mask)
        x = torch.cat([self.upsample2(x), skip2], dim=1)             # (B, c3 + c2, 16, 16)
        for block in self.up2:
            x = block(x, time, context, pad_mask)
        x = torch.cat([self.upsample1(x), skip1], dim=1)             # (B, c2 + c1, 32, 32)
        for block in self.up1:
            x = block(x, time, context, pad_mask)
        if self.outer_level:
            x = torch.cat([self.upsample0(x), skip0], dim=1)         # (B, c1 + c1, 64, 64)
            for block in self.up0:
                x = block(x, time, context, pad_mask)
        return self.conv_out(x)
