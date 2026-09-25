"""Text encoder: from the ids of a caption to one vector per token, the condition of the UNet."""

import torch
from torch import nn

from models.attention import build_attention
from models.sinusoidal import sinusoidal_embedding


class EncoderBlock(nn.Module):
    """One Transformer encoder block: self-attention and feed-forward, each followed by add & norm."""

    def __init__(self, config):
        """Args:
            config: the project Config (uses D_MODEL, FFN_DIM, DROPOUT and the attention settings).
        """
        super().__init__()
        self.attention = build_attention(config, config.D_MODEL)
        self.norm1 = nn.LayerNorm(config.D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.D_MODEL, config.FFN_DIM),
            nn.ReLU(),
            nn.Linear(config.FFN_DIM, config.D_MODEL),
        )
        self.norm2 = nn.LayerNorm(config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x, pad_mask):
        """Args:
            x: (B, L, D), one vector per token.
            pad_mask: (B, L) bool, True where the token is <pad>.

        Returns:
            (B, L, D): every vector now also carries information from the other tokens.
        """
        attended, _ = self.attention(x, x, x, key_padding_mask=pad_mask)   # self-attention: q, k, v all from x
        x = self.norm1(x + self.dropout(attended))                # add & norm
        x = self.norm2(x + self.dropout(self.feed_forward(x)))    # add & norm
        return x


class TextEncoder(nn.Module):
    """Embeddings + positional encoding + NUM_ENCODER_BLOCKS encoder blocks.

    Trained from scratch together with the UNet.
    """

    def __init__(self, config, vocabulary):
        """Args:
            config: the project Config (uses D_MODEL, DROPOUT, NUM_ENCODER_BLOCKS and the block settings).
            vocabulary: the loaded Vocabulary (gives the number of tokens, max_length and the <pad> id).
        """
        super().__init__()
        self.pad_id = vocabulary.ids['<pad>']
        # padding_idx: the <pad> vector stays zero and is never trained
        self.embedding = nn.Embedding(len(vocabulary.tokens), config.D_MODEL, padding_idx=self.pad_id)
        # buffer: not trained, but it follows the model to the GPU and into the saved weights
        positions = torch.arange(vocabulary.max_length)
        self.register_buffer('positional', sinusoidal_embedding(positions, config.D_MODEL))   # (L, D)
        self.dropout = nn.Dropout(config.DROPOUT)
        # ModuleList, not a Python list: otherwise the weights of the blocks would not be registered
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.NUM_ENCODER_BLOCKS)])

    def forward(self, tokens):
        """Args:
            tokens: (B, L) caption ids from Vocabulary.encode, L = max_length.

        Returns:
            (context (B, L, D) one vector per token, pad_mask (B, L) True where the token is <pad>).
        """
        pad_mask = tokens == self.pad_id
        # No * sqrt(D_MODEL) as in the paper: nn.Embedding starts around +-1 like sin and cos,
        # multiplying by sqrt(128) ~ 11 would drown the position.
        x = self.dropout(self.embedding(tokens) + self.positional)   # the same (L, D) added to every caption
        for block in self.blocks:
            x = block(x, pad_mask)
        return x, pad_mask
