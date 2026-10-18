"""Trains the diffusion model (text encoder + UNet).

With --smoke-test it trains only on the first images of the training split, to check that the model can learn them.
The settings are in config.py, the results (last.pt, log.csv, control grids) go to runs/<experiment>/.
If runs/<experiment>/last.pt exists, the training resumes from it.
"""

import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from models.sampling import sample
from preprocessing.cartoon_dataset import CartoonDataset
from preprocessing.image_preprocessor import ImagePreprocessor


def diffusion_noise(shape, offset_noise, generator=None, device='cpu'):
    """The noise put into the images, the plain noise plus the offset noise if OFFSET_NOISE > 0.

    With only the plain noise (one random number per pixel) the mean color of the image stays visible, so the
    model never learns to choose it and paints grey backgrounds. The offset adds one random number per channel,
    the same on all the pixels, and hides the mean color too.

    Args:
        shape: (B, 3, H, W), with B = images, 3 = RGB colors, H = W = side of the images.
        offset_noise: strength of the offset (OFFSET_NOISE); 0 = only the plain noise.
        generator: torch.Generator of the random numbers (the validation uses a fixed seed), or None.
        device: where the noise is created.

    Returns:
        (B, 3, H, W) noise.
    """
    noise = torch.randn(shape, generator=generator, device=device)       # eps, one Gaussian number per pixel
    if offset_noise:
        # (B, 3, 1, 1), one number per channel of every image, added to all its pixels. It is drawn after eps and
        # only when used, so OFFSET_NOISE = 0 gives exactly the random numbers of the plain noise
        offset = torch.randn(shape[0], shape[1], 1, 1, generator=generator, device=device)
        noise = noise + offset_noise * offset
    return noise


def diffusion_loss(model, scheduler, images, tokens, generator=None, offset_noise=0.0):
    """Draw a random step t for every image, add the noise of that step and compare it with the noise predicted
    by the model.

    Args:
        model: DiffusionModel.
        scheduler: NoiseScheduler.
        images: (B, 3, H, W) clean images in [-1, 1], with B = images, 3 = RGB colors, H = W = side of the images.
        tokens: (B, L) caption ids, with L = tokens per caption.
        generator: torch.Generator for t and the noise (the validation uses a fixed seed), or None.
        offset_noise: strength of the offset noise (OFFSET_NOISE), see diffusion_noise; 0 = no offset noise.

    Returns:
        Tensor with one number, the mean squared error between the predicted and the real noise.
    """
    device = images.device

    # a random step from 0 to T - 1 for every image
    t = torch.randint(0, scheduler.num_timesteps, (len(images),), generator=generator, device=device)
    noise = diffusion_noise(images.shape, offset_noise, generator, device)                             # eps (+ offset)
    x_t = scheduler.add_noise(images, t, noise)           # sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps

    return F.mse_loss(model(x_t, t, tokens), noise)       # || eps - eps_theta(x_t, t, text) ||^2


def update_ema(ema, model, step, config):
    """Move every EMA weight a little towards the weight of the model, with ema = decay * ema + (1 - decay) * weight.

    The decay grows from 0.1 to EMA_DECAY in the first steps, because with 0.999 from the start the EMA would still
    be mostly the random initial weights for about the first 1000 steps.

    Args:
        ema: the EMA copy of the model.
        model: the model being trained.
        step: training steps done before this one (0 at the first step).
        config: the project Config (uses EMA_DECAY).
    """
    decay = min(config.EMA_DECAY, (1 + step) / (10 + step))

    with torch.no_grad():
        for ema_weight, weight in zip(ema.parameters(), model.parameters()):
            ema_weight.mul_(decay).add_(weight, alpha=1 - decay)


def save_checkpoint(checkpoint, path):
    """Write the checkpoint to a temporary file, then put it in place of path in one operation.

    An interruption while writing leaves the previous checkpoint intact, so the training can still resume.

    Args:
        checkpoint: dictionary to save.
        path: destination file (last.pt).
    """
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(config, smoke_test=False):
    """Train, validate, save the control grids and the checkpoint (resume from last.pt if it exists).

    Args:
        config: the project Config (uses the Training and Smoke test sections and the settings of the model).
        smoke_test: train only on the first SMOKE_TEST_IMAGES training images, with the smoke test settings,
            to check that the model can learn them by heart.
    """
    print('Training of the cartoon diffusion model' + (' (smoke test)' if smoke_test else ''))
    if smoke_test:
        # the smoke test settings replace the normal ones only in this Config object, config.py does not change
        config.EPOCHS = config.SMOKE_TEST_EPOCHS
        config.SAMPLE_EVERY = config.SMOKE_TEST_SAMPLE_EVERY
        config.CHECKPOINT_EVERY = config.SMOKE_TEST_CHECKPOINT_EVERY

    torch.manual_seed(config.SEED)
    if config.DETERMINISTIC:
        # the GPU always adds up in the same order, so two trainings from the same seed give the same model
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    # required by cuBLAS for deterministic results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('No GPU found')

    # 1. The folder of this experiment, named after its settings, so different experiments never overwrite each other
    name = 'conditional' if config.TEXT_CONDITIONING else 'unconditional'
    if smoke_test:
        name += '_smoke_test'
    name += f'_{config.IMAGE_SIZE}'     # the size at the end, e.g. runs/conditional_32 or runs/conditional_64
    run_dir = config.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'1. Experiment: {run_dir} (device: {device})')

    # 2. The data prepared by prepare_data.py (the smoke test reads only its first images from the disk)
    train = CartoonDataset(config, 'train', limit=config.SMOKE_TEST_IMAGES if smoke_test else None)
    vocabulary = train.vocabulary
    val = CartoonDataset(config, 'val')
    train_loader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=config.BATCH_SIZE)
    print(f'2. Data: train {len(train)} images, val {len(val)} images')

    # 3. The model, its EMA copy (to validate and to generate), the noise recipe, the optimizer
    model = DiffusionModel(config, vocabulary).to(device)
    ema = copy.deepcopy(model).eval()
    scheduler = NoiseScheduler(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    print(f'3. Model: {sum(p.numel() for p in model.parameters()):,} parameters')

    # 4. Resume from the last checkpoint, if there is one
    checkpoint_file = run_dir / 'last.pt'
    first_epoch = 1
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        if device == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        first_epoch = checkpoint['epoch'] + 1
        print(f'4. Resumed after epoch {checkpoint["epoch"]}')
    else:
        print('4. New training')

    # the settings of this run, as plain values (paths become text)
    settings = json.loads(json.dumps({name: getattr(config, name) for name in dir(config) if name.isupper()},
                                     default=str))

    # the 8 captions of the control grid, each one twice, so every caption gets two different images
    real_images = []    # only in the smoke test, the real images shown in the first row of the grid
    if smoke_test:
        first = [train[i] for i in range(8)]    # the first 8 training images and their captions
        real_images = [ImagePreprocessor.denormalize(image) for image, _ in first]
        grid_tokens = torch.stack([tokens for _, tokens in first] * 2).to(device)
    else:
        grid_tokens = torch.tensor([vocabulary.encode(prompt) for prompt in config.SAMPLE_PROMPTS] * 2, device=device)

    step = (first_epoch - 1) * len(train_loader)     # training steps already done (for the EMA)

    print(f'5. Epochs {first_epoch}-{config.EPOCHS}')
    start = time.time()
    total, steps = 0.0, 0                            # training loss since the last row of the log
    for epoch in range(first_epoch, config.EPOCHS + 1):
        model.train()
        for images, tokens in train_loader:
            images, tokens = images.to(device), tokens.to(device)
            if config.TEXT_CONDITIONING:
                # for classifier-free guidance a portion of the captions becomes the empty caption
                drop = torch.rand(len(tokens), device=device) < config.CAPTION_DROPOUT
                tokens = torch.where(drop[:, None], model.empty_tokens, tokens)

            loss = diffusion_loss(model, scheduler, images, tokens, offset_noise=config.OFFSET_NOISE)
            optimizer.zero_grad()
            loss.backward()                                                      # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()                                                     # AdamW updates the weights

            update_ema(ema, model, step, config)
            step += 1
            total += loss.item()
            steps += 1

        # validation, checkpoint and log only every CHECKPOINT_EVERY epochs (and at the last one), because in the
        # smoke test an epoch is a single step and doing them every time would take most of the time
        checkpoint_epoch = epoch % config.CHECKPOINT_EVERY == 0 or epoch == config.EPOCHS
        if checkpoint_epoch:
            # validation loss of the EMA model on the real captions. The generator starts again from SEED every
            # time, so t and the noise are the same at every validation and the loss changes only with the model
            generator = torch.Generator(device=device).manual_seed(config.SEED)
            with torch.no_grad():
                val_loss = sum(diffusion_loss(ema, scheduler, images.to(device), tokens.to(device), generator,
                                              config.OFFSET_NOISE).item()
                               for images, tokens in val_loader) / len(val_loader)

        if epoch % config.SAMPLE_EVERY == 0:
            # control grid with the same captions and seed every time, so only the model changes
            images = sample(ema, scheduler, grid_tokens, config, config.SEED)
            grid_file = run_dir / f'samples_epoch{epoch:03d}.png'
            generated = [ImagePreprocessor.denormalize(image) for image in images]
            ImagePreprocessor(config).save_preview(real_images + generated, grid_file)
            print(f'   control grid: {grid_file.name}')

        if checkpoint_epoch:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                'settings': settings,
            }, checkpoint_file)

            # the row of the log is written after the checkpoint, so an interruption in between repeats epochs but
            # never rows
            train_loss, seconds = total / steps, time.time() - start
            log_file = run_dir / 'log.csv'
            new_log = not log_file.exists()
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_log:
                    writer.writerow(['epoch', 'train_loss', 'val_loss', 'seconds'])
                writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{seconds:.1f}'])

            print(f'   epoch {epoch}/{config.EPOCHS}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, {seconds:.0f} s')
            start = time.time()
            total, steps = 0.0, 0

    print(f'Done: {run_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the cartoon diffusion model (settings in config.py).')
    parser.add_argument('--smoke-test', action='store_true',
                        help='train only on the first SMOKE_TEST_IMAGES training images, to check that it learns them')
    main(Config(), smoke_test=parser.parse_args().smoke_test)
