"""Tests of the text encoder and its parts. Run them with:  python -m pytest tests/"""

import pytest
import torch
from torch import nn

from config import Config
from models.attention import MultiHeadAttention, build_attention
from models.sinusoidal import sinusoidal_embedding
from models.text_encoder import TextEncoder
from preprocessing.vocabulary import Vocabulary

# two captions of different length: the second one ends with one <pad>
CAPTIONS = [
    'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard',
    'a cartoon avatar with dark skin, short black hair, sunglasses and no beard',
]


def make_vocabulary(config):
    """A small vocabulary built from CAPTIONS, so the tests do not need data/."""
    vocabulary = Vocabulary(config)
    vocabulary.build(CAPTIONS)
    return vocabulary


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


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    ours = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.1)
    theirs = nn.MultiheadAttention(16, 4, dropout=0.1, batch_first=True)
    # same weights in both: PyTorch keeps q, k and v stacked in one matrix
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        theirs.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        theirs.out_proj.weight.copy_(ours.w_o.weight)
        theirs.out_proj.bias.copy_(ours.w_o.bias)
    ours.eval()     # no dropout: the two must give the same numbers
    theirs.eval()

    x = torch.randn(2, 5, 16)
    pad_mask = torch.tensor([[False, False, False, False, True],
                             [False, False, False, True, True]])
    our_output, our_weights = ours(x, x, x, key_padding_mask=pad_mask)
    their_output, their_weights = theirs(x, x, x, key_padding_mask=pad_mask)
    assert torch.allclose(our_output, their_output, atol=1e-5)
    assert torch.allclose(our_weights, their_weights, atol=1e-5)
    assert torch.all(our_weights[:, :, 4] == 0)   # nobody looks at a <pad>


def test_attention_with_text_of_different_size_matches_pytorch():
    torch.manual_seed(0)
    # queries of 16 numbers (the image), keys and values from vectors of 8 numbers (the text)
    ours = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.1, kdim=8)
    theirs = nn.MultiheadAttention(16, 4, dropout=0.1, batch_first=True, kdim=8, vdim=8)
    # with kdim PyTorch keeps W_Q, W_K, W_V in three matrices (the biases stay stacked in one)
    with torch.no_grad():
        theirs.q_proj_weight.copy_(ours.w_q.weight)
        theirs.k_proj_weight.copy_(ours.w_k.weight)
        theirs.v_proj_weight.copy_(ours.w_v.weight)
        theirs.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        theirs.out_proj.weight.copy_(ours.w_o.weight)
        theirs.out_proj.bias.copy_(ours.w_o.bias)
    ours.eval()     # no dropout: the two must give the same numbers
    theirs.eval()

    pixels = torch.randn(2, 6, 16)   # 6 pixel tokens
    words = torch.randn(2, 5, 8)     # 5 word tokens
    pad_mask = torch.tensor([[False, False, False, False, True],
                             [False, False, False, True, True]])
    our_output, our_weights = ours(pixels, words, words, key_padding_mask=pad_mask)
    their_output, their_weights = theirs(pixels, words, words, key_padding_mask=pad_mask)
    assert our_output.shape == (2, 6, 16)
    assert torch.allclose(our_output, their_output, atol=1e-5)
    assert torch.allclose(our_weights, their_weights, atol=1e-5)


def test_attention_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=128, num_heads=3, dropout=0.1)


def test_build_attention():
    config = Config()
    config.ATTENTION = 'scratch'
    assert isinstance(build_attention(config, config.D_MODEL), MultiHeadAttention)
    config.ATTENTION = 'torch'
    assert isinstance(build_attention(config, config.D_MODEL), nn.MultiheadAttention)
    config.ATTENTION = 'Scratch'   # a typo must stop, not silently pick one of the two
    with pytest.raises(ValueError):
        build_attention(config, config.D_MODEL)


@pytest.mark.parametrize('attention', ['scratch', 'torch'])
def test_text_encoder(attention):
    config = Config()
    config.ATTENTION = attention
    vocabulary = make_vocabulary(config)
    encoder = TextEncoder(config, vocabulary)
    tokens = torch.tensor([vocabulary.encode(caption) for caption in CAPTIONS])   # (2, 18)

    context, pad_mask = encoder(tokens)
    assert context.shape == (2, 18, config.D_MODEL)
    assert torch.equal(pad_mask, tokens == vocabulary.ids['<pad>'])
    assert pad_mask[1, -1] and not pad_mask[0].any()   # only the shorter caption has a <pad>

    context, pad_mask = encoder(tokens[:1])   # one prompt alone, as at generation time
    assert context.shape == (1, 18, config.D_MODEL)


def test_text_encoder_trains_every_weight():
    config = Config()
    vocabulary = make_vocabulary(config)
    encoder = TextEncoder(config, vocabulary)
    tokens = torch.tensor([vocabulary.encode(caption) for caption in CAPTIONS])

    context, _ = encoder(tokens)
    (context * torch.randn_like(context)).sum().backward()   # any loss that uses every output
    # the weights of the last block are among the parameters (a Python list would hide them)
    assert f'blocks.{config.NUM_ENCODER_BLOCKS - 1}.feed_forward.0.weight' in dict(encoder.named_parameters())
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, f'{name} is not trained'
    # the positional encoding follows the model (GPU, saved weights) but is not trained
    assert 'positional' in encoder.state_dict()
    assert 'positional' not in dict(encoder.named_parameters())
