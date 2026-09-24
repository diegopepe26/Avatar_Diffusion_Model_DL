"""Multi-head attention written from scratch, interchangeable with nn.MultiheadAttention."""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with several heads.

    Same interface as nn.MultiheadAttention(..., batch_first=True): same arguments of forward,
    same outputs, same mask convention. So build_attention can give either of the two.
    """

    def __init__(self, d_model, num_heads, dropout):
        """Args:
            d_model: size of the input and output vectors (D_MODEL).
            num_heads: number of heads; each one works on d_model / num_heads values.
            dropout: probability of dropping an attention weight during training.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by num_heads ({num_heads})')
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # the weight matrices W_Q, W_K, W_V and W_O of the paper (each Linear also adds a bias)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, key_padding_mask=None):
        """Every query vector collects the value vectors, weighted by how much its key matches.

        x_q, x_k, x_v are the sequences the queries, keys and values are computed from:
        - text encoder (self-attention): all three are the same x, the text;
        - UNet (cross-attention): x_q is the image, x_k and x_v are the text.

        Args:
            x_q: (B, L_q, D), the sequence the queries come from.
            x_k: (B, L_k, D), the sequence the keys come from.
            x_v: (B, L_k, D), the sequence the values come from.
            key_padding_mask: (B, L_k) bool, True where the token is <pad> (ignored); None = no mask.

        Returns:
            (output (B, L_q, D), attention weights averaged over the heads (B, L_q, L_k)).
        """
        batch, query_length, d_model = x_q.shape
        key_length = x_k.shape[1]
        # q, k, v: the queries, keys and values, i.e. x_q, x_k, x_v multiplied by W_Q, W_K, W_V.
        # (B, L, D) -> (B, L, heads, head_dim) -> (B, heads, L, head_dim): each head gets its own slice
        q = self.w_q(x_q).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x_k).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x_v).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)

        # how much each query matches each key; / sqrt(head_dim) keeps the softmax from saturating
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)    # (B, heads, L_q, L_k)
        if key_padding_mask is not None:
            # -inf becomes weight 0 after the softmax: nobody looks at the <pad> tokens
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        # dropout on the weights, where nn.MultiheadAttention puts it
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = weights @ v                                            # (B, heads, L_q, head_dim)

        # back to (B, L_q, D): the heads side by side
        output = output.transpose(1, 2).reshape(batch, query_length, d_model)
        return self.w_o(output), weights.mean(dim=1)


def build_attention(config):
    """Give the attention chosen in config.ATTENTION: ours or the one of PyTorch.

    Args:
        config: the project Config (uses ATTENTION, D_MODEL, NUM_HEADS, DROPOUT).

    Returns:
        MultiHeadAttention ('scratch') or nn.MultiheadAttention ('torch'), called the same way.
    """
    if config.ATTENTION == 'scratch':
        return MultiHeadAttention(config.D_MODEL, config.NUM_HEADS, config.DROPOUT)
    if config.ATTENTION == 'torch':
        # batch_first: inputs (B, L, D) like ours, not the default (L, B, D)
        return nn.MultiheadAttention(config.D_MODEL, config.NUM_HEADS, dropout=config.DROPOUT, batch_first=True)
    raise ValueError(f"ATTENTION must be 'scratch' or 'torch', not '{config.ATTENTION}'")
