"""Generation: from pure noise to images, one step back at a time (DDPM, Algorithm 2)."""

import torch


@torch.no_grad()   # nothing is learned here: no gradients, less memory
def sample(model, scheduler, tokens, config, seed):
    """Generate one image per caption, from pure noise (t = T - 1) back to t = 0.

    With a text encoder every step calls the model twice, with the caption and with the empty caption,
    and pushes the prediction towards the caption (classifier-free guidance):
    noise = without caption + GUIDANCE_SCALE * (with caption - without caption).

    Args:
        model: DiffusionModel, in eval() (use the EMA weights).
        scheduler: NoiseScheduler.
        tokens: (B, L) caption ids, on the device of the model; the unconditional model only uses B.
        config: the project Config (uses IMAGE_SIZE, GUIDANCE_SCALE).
        seed: the same seed and captions always give the same images.

    Returns:
        (B, 3, IMAGE_SIZE, IMAGE_SIZE) generated images, values in [-1, 1].
    """
    device = tokens.device
    batch = len(tokens)
    generator = torch.Generator(device=device).manual_seed(seed)
    # x_T: pure noise
    x = torch.randn(batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, generator=generator, device=device)
    empty = model.empty_tokens.expand(batch, -1)             # the empty caption, once per image
    for t in reversed(range(scheduler.num_timesteps)):       # t = T - 1, ..., 1, 0
        steps = torch.full((batch,), t, device=device)
        if model.text_encoder is None:
            predicted = model(x, steps, tokens)
        else:
            with_caption = model(x, steps, tokens)
            without_caption = model(x, steps, empty)
            predicted = without_caption + config.GUIDANCE_SCALE * (with_caption - without_caption)
        x = scheduler.step(x, t, predicted, generator)
    return x
