"""Tests of the text encoder and its parts. Run them with:  python -m pytest tests/"""

import pytest
import torch

from models.sinusoidal import sinusoidal_embedding


def test_sinusoidal_embedding():
    embedding = sinusoidal_embedding(torch.arange(18), 128)
    assert embedding.shape == (18, 128)
    # position 0: every angle is 0, so sin = 0 (even columns) and cos = 1 (odd columns)
    assert torch.all(embedding[0, 0::2] == 0)
    assert torch.all(embedding[0, 1::2] == 1)
    # the first frequency is 1: position 5 gives sin(5) and cos(5)
    assert torch.isclose(embedding[5, 0], torch.sin(torch.tensor(5.0)))
    assert torch.isclose(embedding[5, 1], torch.cos(torch.tensor(5.0)))


def test_sinusoidal_embedding_odd_dim():
    with pytest.raises(ValueError):
        sinusoidal_embedding(torch.arange(18), 127)
