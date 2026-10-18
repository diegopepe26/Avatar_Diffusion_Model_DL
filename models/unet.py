"""UNet denoiser that predicts the noise added to x_t, given the step t and, optionally, the caption."""

import torch
from torch import nn
from torch.nn import functional as F

from models.attention import MultiHeadAttention
from models.sinusoidal import sinusoidal_embedding


class ResBlock(nn.Module):
    """Residual block composed by two 3x3 convolutions with the time embedding added in between."""

    def __init__(self, in_channels, out_channels, config, film=False):
        """Args:
            in_channels: channels of the input.
            out_channels: channels of the output.
            config: the project Config (uses TIME_DIM, DROPOUT and, with film, D_MODEL).
            film: True in the pooled ablation on the blocks that would have the cross-attention, where the caption
                vector scales and shifts every channel (FiLM).
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)      # 32 groups
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # brings the time embedding to one value per channel
        self.time_projection = nn.Linear(config.TIME_DIM, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        # FiLM turns the caption vector into a scale gamma and a shift beta for every channel (2 * out_channels)
        self.film = nn.Linear(config.D_MODEL, 2 * out_channels) if film else None
        self.dropout = nn.Dropout(config.DROPOUT)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # the input is added to the output, through a 1x1 convolution when the channels change
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time, text=None):
        """Args:
            x: (B, in_channels, H, W).
            time: (B, TIME_DIM), the time embedding of the step t of every image.

        Returns:
            (B, out_channels, H, W).
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # the same value on every pixel of a channel, [:, :, None, None] makes it (B, C, 1, 1)
        h = h + self.time_projection(F.silu(time))[:, :, None, None]

        h = self.norm2(h)
        if self.film is not None:
            # h * (1 + gamma) + beta. With gamma and beta near 0, as at the start of the
            # training, the channel stays almost as it is, and gamma = -1 switches it off. The same two numbers
            # are used on every pixel
            gamma, beta = self.film(text).chunk(2, dim=1)                  # (B, C) each
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(h)))
        return self.shortcut(x) + h


class CrossAttention(nn.Module):
    """The pixels read the words of the caption, with the queries from the image and the keys and values from the
    text.

    The attention itself is the MultiHeadAttention of attention.py, this block only turns the image into a
    sequence of tokens and back.
    """

    def __init__(self, channels, config):
        """Args:
            channels: channels of the image at this level (the size of the attention).
            config: the project Config (uses D_MODEL, the size of the text vectors, NUM_HEADS, DROPOUT).
        """
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)

        # W_K and W_V take the words (D_MODEL numbers) straight to the channels of the level
        self.attention = MultiHeadAttention(channels, config.NUM_HEADS, config.DROPOUT, kdim=config.D_MODEL)

    def forward(self, x, context, pad_mask):
        """Args:
            x: (B, C, H, W) image features.
            context: (B, L, D_MODEL) the words, from the TextEncoder.
            pad_mask: (B, L) bool, True where the token is <pad>.

        Returns:
            (B, C, H, W), x plus what the pixels read from the text.
        """
        batch, channels, height, width = x.shape

        # every pixel becomes a token, (B, C, H, W) -> (B, H*W, C)
        pixels = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attention(pixels, context, context, key_padding_mask=pad_mask)

        # back to an image, added to the input (residual)
        return x + attended.transpose(1, 2).reshape(batch, channels, height, width)


class UNetBlock(nn.Module):
    """A ResBlock, followed by a CrossAttention on the levels that read the caption (16x16 and 8x8).

    There is no CrossAttention on the other levels, in the unconditional model and in the pooled ablation, where
    the ResBlock of the same levels has FiLM inside instead.
    """

    def __init__(self, in_channels, out_channels, config, attention):
        """Args:
            in_channels: channels of the input.
            out_channels: channels of the output.
            config: the project Config (uses TEXT_CONDITIONING, CONDITIONING and the block settings).
            attention: True on the levels that read the text (16x16 and 8x8).
        """
        super().__init__()
        pooled = config.TEXT_CONDITIONING and config.CONDITIONING == 'pooled'
        self.res = ResBlock(in_channels, out_channels, config, film=attention and pooled)

        # None when the level has no cross-attention, in the unconditional model or with the pooled text
        cross_attention = config.TEXT_CONDITIONING and config.CONDITIONING == 'cross_attention'
        self.attention = CrossAttention(out_channels, config) if attention and cross_attention else None

    def forward(self, x, time, context, pad_mask, text=None):
        """Args:
            x: (B, in_channels, H, W).
            time: (B, TIME_DIM).
            context: (B, L, D_MODEL), or None without text.
            pad_mask: (B, L), or None without text.
            text: (B, D_MODEL) the mean of the caption tokens, only in the pooled ablation.

        Returns:
            (B, out_channels, H, W).
        """
        x = self.res(x, time, text)
        if self.attention is not None:
            x = self.attention(x, context, pad_mask)
        return x


class UNet(nn.Module):
    """Two blocks per level, skip connections by concatenation, the bottom always at 8x8.

    With IMAGE_SIZE = 32 there are three resolutions (32, 16, 8). With IMAGE_SIZE = 64 it is the same network with
    one more level outside, at 64x64 (64, 32, 16, 8).
    """

    def __init__(self, config):
        """Args:
            config: the project Config (uses IMAGE_SIZE, UNET_CHANNELS, TIME_DIM, TEXT_CONDITIONING, CONDITIONING,
                D_MODEL and the block settings).
        """
        super().__init__()

        # channels of the levels, c1 at 32x32 (and at 64x64), c2 at 16x16, c3 at 8x8. They are 64, 128, 256 with
        # IMAGE_SIZE = 32 and 128, 256, 512 with IMAGE_SIZE = 64
        c1, c2, c3 = config.UNET_CHANNELS
        self.base_channels = c1

        # at 64x64 there is one more level outside the 32x32 network (64 -> 32), so the bottom is still 8x8, the
        # cross-attention still at 16x16 and 8x8 and every bottom pixel still sees the whole image. At 32x32
        # these layers do not exist
        self.outer_level = config.IMAGE_SIZE == 64

        # the time embedding is sin/cos of t (the same formula as the positions of the text), then a small MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(c1, config.TIME_DIM),
            nn.SiLU(),
            nn.Linear(config.TIME_DIM, config.TIME_DIM),
        )

        # in the pooled ablation the caption becomes one vector of D_MODEL numbers, the mean of its tokens, that
        # scales and shifts the channels (FiLM) of the blocks that would have the cross-attention. With
        # 'cross_attention' the FiLM layers do not exist
        self.pooled = config.TEXT_CONDITIONING and config.CONDITIONING == 'pooled'

        self.conv_in = nn.Conv2d(3, c1, 3, padding=1)

        # going down, two blocks per level, then a stride-2 convolution halves the size
        if self.outer_level:
            self.down0 = nn.ModuleList([UNetBlock(c1, c1, config, False), UNetBlock(c1, c1, config, False)])
            self.downsample0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                      # 64 -> 32
        self.down1 = nn.ModuleList([UNetBlock(c1, c1, config, False), UNetBlock(c1, c1, config, False)])
        self.downsample1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)                          # 32 -> 16
        self.down2 = nn.ModuleList([UNetBlock(c1, c2, config, True), UNetBlock(c2, c2, config, True)])
        self.downsample2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)                          # 16 -> 8

        # at the bottom ResBlock -> CrossAttention -> ResBlock
        self.middle = nn.ModuleList([UNetBlock(c2, c3, config, True), UNetBlock(c3, c3, config, False)])

        # going up, nearest interpolation + 3x3 convolution, then the skip is concatenated
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
            context: (B, L, D_MODEL) from the TextEncoder. Not needed if TEXT_CONDITIONING is False.
            pad_mask: (B, L) bool from the TextEncoder, True where the token is <pad>.

        Returns:
            (B, 3, IMAGE_SIZE, IMAGE_SIZE) predicted noise.
        """
        time =self.time_mlp(sinusoidal_embedding(t, self.base_channels))    # (B, TIME_DIM)

        text = None
        if self.pooled:
            # mean of the real tokens, <pad> excluded (the empty caption keeps <bos> and <eos>, never zero tokens)
            real = (~pad_mask).unsqueeze(-1).float()                          # (B, L, 1), 1 on the real tokens
            text = (context * real).sum(dim=1) / real.sum(dim=1)              # (B, D_MODEL), to the FiLM blocks

        x = self.conv_in(x_t)

        if self.outer_level:
            for block in self.down0:
                x = block(x, time, context, pad_mask, text)
            skip0 = x                                                # (B, c1, 64, 64)
            x = self.downsample0(x)
        for block in self.down1:
            x = block(x, time, context, pad_mask, text)
        skip1 = x                                                    # (B, c1, 32, 32)
        x = self.downsample1(x)
        for block in self.down2:
            x = block(x, time, context, pad_mask, text)
        skip2 = x                                                    # (B, c2, 16, 16)
        x = self.downsample2(x)

        for block in self.middle:
            x = block(x, time, context, pad_mask, text)

        x = torch.cat([self.upsample2(x), skip2], dim=1)             # (B, c3 + c2, 16, 16)
        for block in self.up2:
            x = block(x, time, context, pad_mask, text)
        x = torch.cat([self.upsample1(x), skip1], dim=1)             # (B, c2 + c1, 32, 32)
        for block in self.up1:
            x = block(x, time, context, pad_mask, text)
        if self.outer_level:
            x = torch.cat([self.upsample0(x), skip0], dim=1)         # (B, c1 + c1, 64, 64)
            for block in self.up0:
                x = block(x, time, context, pad_mask, text)

        return self.conv_out(x)
