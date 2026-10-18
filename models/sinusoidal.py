"""Sinusoidal embedding of the token positions in the text encoder and of the time steps in the UNet."""

import torch


def sinusoidal_embedding(positions, dim):
    """Turn each position into a vector of sines and cosines at different frequencies.

    PE(pos, 2i) = sin(pos / 10000^(2i/dim)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/dim)).
    Nothing is trained, the same position always gives the same vector.

    Args:
        positions: tensor (N,), e.g. the token positions 0..17 or the time steps t of a batch.
        dim: size of each vector (even), e.g. D_MODEL.

    Returns:
        Tensor (N, dim), on the same device as positions.
    """
    two_i = torch.arange(0, dim, 2, device=positions.device)      # 0, 2, 4, ..., dim - 2
    frequencies = 1 / 10000 ** (two_i / dim)                       # (dim/2,)
    angles = positions.float()[:, None] * frequencies[None, :]     # (N, dim/2)

    embedding = torch.zeros(len(positions), dim, device=positions.device)
    embedding[:, 0::2] = torch.sin(angles)   # even columns
    embedding[:, 1::2] = torch.cos(angles)   # odd columns

    return embedding
