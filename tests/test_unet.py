"""Tests of the UNet. Run them with:  python -m pytest tests/"""

import pytest
import torch

from config import Config
from models.text_encoder import TextEncoder
from models.unet import CrossAttention, UNet
from preprocessing.vocabulary import Vocabulary


def make_inputs(batch=2, size=32, text_dim=128):
    """Random noisy images, steps and text, shaped like the real ones (18 tokens, the last one <pad>)."""
    torch.manual_seed(0)
    x_t = torch.randn(batch, 3, size, size)
    t = torch.randint(0, 1000, (batch,))
    context = torch.randn(batch, 18, text_dim)
    pad_mask = torch.zeros(batch, 18, dtype=torch.bool)
    pad_mask[:, -1] = True
    return x_t, t, context, pad_mask


@pytest.mark.parametrize('attention', ['scratch', 'torch'])
def test_unet_output_shape(attention):
    config = Config()
    config.ATTENTION = attention
    unet = UNet(config)
    x_t, t, context, pad_mask = make_inputs()
    assert unet(x_t, t, context, pad_mask).shape == (2, 3, 32, 32)   # the predicted noise, like x_t


def test_unet_32_keeps_its_layers():
    # the same layers as before the 64x64 level: the checkpoints of the 32x32 trainings still load
    layers = {name.split('.')[0] for name in UNet(Config()).state_dict()}
    assert layers == {'time_mlp', 'conv_in', 'down1', 'downsample1', 'down2', 'downsample2', 'middle',
                      'upsample2', 'up2', 'upsample1', 'up1', 'conv_out'}


def test_unet_64_has_one_more_level():
    config = Config()
    config.IMAGE_SIZE = 64
    unet = UNet(config)
    sizes = []
    unet.middle[0].register_forward_pre_hook(lambda module, inputs: sizes.append(inputs[0].shape[-1]))
    x_t, t, context, pad_mask = make_inputs(size=64)
    assert unet(x_t, t, context, pad_mask).shape == (2, 3, 64, 64)
    assert sizes == [8]                                              # the bottom is still 8x8
    assert sum(isinstance(module, CrossAttention) for module in unet.modules()) == 5
    assert all(block.attention is None for block in [*unet.down0, *unet.up0])   # no text at 64x64


def test_unet_image_size_must_be_32_or_64():
    config = Config()
    config.IMAGE_SIZE = 48
    with pytest.raises(ValueError):
        UNet(config)


def test_unet_with_d_model_64():
    config = Config()
    config.D_MODEL = 64                                              # text vectors of 64 numbers
    unet = UNet(config)
    x_t, t, context, pad_mask = make_inputs(text_dim=64)
    assert unet(x_t, t, context, pad_mask).shape == (2, 3, 32, 32)


def test_unconditional_unet():
    config = Config()
    config.TEXT_CONDITIONING = False
    unet = UNet(config)
    x_t, t, _, _ = make_inputs()
    assert unet(x_t, t).shape == (2, 3, 32, 32)                      # no text at all
    assert not any(isinstance(module, CrossAttention) for module in unet.modules())
    conditional = UNet(Config())
    assert sum(isinstance(module, CrossAttention) for module in conditional.modules()) == 5


def test_text_and_time_change_the_output():
    unet = UNet(Config()).eval()                                     # eval: no dropout
    x_t, t, context, pad_mask = make_inputs()
    other_context = torch.randn_like(context)
    changed_pad = context.clone()
    changed_pad[:, -1] = torch.randn(2, 128)                         # only the <pad> position changes
    with torch.no_grad():
        output = unet(x_t, t, context, pad_mask)
        assert not torch.allclose(output, unet(x_t, t, other_context, pad_mask))            # the text is used
        assert not torch.allclose(output, unet(x_t, (t + 500) % 1000, context, pad_mask))   # t is used
        assert torch.allclose(output, unet(x_t, t, changed_pad, pad_mask), atol=1e-5)       # <pad> is ignored


def test_conditional_unet_needs_the_context():
    unet = UNet(Config())
    x_t, t, _, _ = make_inputs()
    with pytest.raises(ValueError):
        unet(x_t, t)


def test_unet_trains_every_weight():
    unet = UNet(Config())
    x_t, t, context, pad_mask = make_inputs()
    output = unet(x_t, t, context, pad_mask)
    (output * torch.randn_like(output)).sum().backward()   # any loss that uses every output
    for name, parameter in unet.named_parameters():
        assert parameter.grad is not None, f'{name} is not trained'


def test_gradient_reaches_the_text_encoder():
    caption = 'a cartoon avatar with pale skin, long blonde hair, no glasses and a beard'
    config = Config()
    vocabulary = Vocabulary(config)
    vocabulary.build([caption])
    text_encoder = TextEncoder(config, vocabulary)
    unet = UNet(config)
    tokens = torch.tensor([vocabulary.encode(caption)] * 2)
    x_t, t, _, _ = make_inputs()

    context, pad_mask = text_encoder(tokens)
    output = unet(x_t, t, context, pad_mask)
    (output * torch.randn_like(output)).sum().backward()   # any loss on the output of the UNet
    # the loss of the UNet must train the text encoder too, down to the vectors of the words
    gradient = text_encoder.embedding.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_unet_refuses_images_of_the_wrong_size():
    config = Config()
    config.IMAGE_SIZE = 64
    unet = UNet(config)
    x_t, t, context, pad_mask = make_inputs(size=32)   # e.g. 32x32 images while IMAGE_SIZE is 64
    with pytest.raises(ValueError):
        unet(x_t, t, context, pad_mask)
