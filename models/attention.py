"""Multi-head attention written from scratch."""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with several heads, used as self-attention in the text encoder and as
    cross-attention in the UNet."""

    def __init__(self, d_model, num_heads, dropout, kdim=None):
        """Args:
            d_model: size of the queries and of the output (D_MODEL in the text encoder, the channels of the level
                in the UNet).
            num_heads: number of heads, each one works on d_model / num_heads values.
            dropout: probability of dropping an attention weight during training.
            kdim: numbers per token of x_k and x_v, if different from d_model (the text in the UNet).
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # the weight matrices W_Q, W_K, W_V and W_O, each Linear also adds a bias.
        # W_K and W_V take kdim numbers per token and give d_model, so the text can be smaller than the image
        kdim = d_model if kdim is None else kdim
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(kdim, d_model)
        self.w_v = nn.Linear(kdim, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, key_padding_mask=None):
        """Compute the attention scores in every head, then concatenate the heads and multiply by W_O.

        x_q, x_k, x_v are the sequences the queries, keys and values are computed from. In the text encoder
        (self-attention) all three are the same x, the text. In the UNet (cross-attention) x_q is the image and
        x_k, x_v are the text.

        Args:
            x_q: (B, L_q, D), the sequence the queries come from.
            x_k: (B, L_k, kdim), the sequence the keys come from.
            x_v: (B, L_k, kdim), the sequence the values come from.
            key_padding_mask: (B, L_k) bool, True where the token is <pad> (ignored); None = no mask.

        Returns:
            (B, L_q, D), one output vector per query.
        """
        batch, query_length, d_model = x_q.shape
        key_length = x_k.shape[1]

        # the queries, keys and values are x_q, x_k, x_v multiplied by W_Q, W_K, W_V.
        # (B, L, D) -> (B, L, heads, head_dim) -> (B, heads, L, head_dim), so each head gets its own slice
        q = self.w_q(x_q).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x_k).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x_v).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)

        # how much each query matches each key, divided by sqrt(head_dim) so the softmax does not saturate
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)    # (B, heads, L_q, L_k)
        if key_padding_mask is not None:
            # -inf becomes weight 0 after the softmax, so no token looks at the <pad> tokens
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        # the dropout on the attention weights switches off some links between tokens during training
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = weights @ v                                            # (B, heads, L_q, head_dim)

        # back to (B, L_q, D) with the heads side by side
        output = output.transpose(1, 2).reshape(batch, query_length, d_model)

        return self.w_o(output)
