"""Trains the diffusion model (text encoder + UNet) with the training algorithm of DDPM.

Run it with:  python train.py      (after prepare_data.py, which prepares data/)
The settings are in config.py; the results go to runs/<experiment>/: last.pt, log.csv, control grids.
If runs/<experiment>/last.pt exists, the training resumes from it.
"""

import copy
import csv
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import Config
from models.diffusion import DiffusionModel
from models.noise_scheduler import NoiseScheduler
from models.sampling import sample
from preprocessing.cartoon_dataset import CartoonDataset
from preprocessing.image_preprocessor import ImagePreprocessor


def diffusion_loss(model, scheduler, images, tokens, generator=None):
    """The diffusion objective (DDPM, Algorithm 1): how well the model guesses the noise put into the images.

    Args:
        model: DiffusionModel.
        scheduler: NoiseScheduler.
        images: (B, 3, H, W) clean images in [-1, 1].
        tokens: (B, L) caption ids.
        generator: torch.Generator for t and the noise (the validation uses a fixed seed), or None.

    Returns:
        Tensor with one number: mean squared error between the predicted and the real noise.
    """
    device = images.device
    t = torch.randint(0, scheduler.num_timesteps, (len(images),), generator=generator, device=device)  # t ~ U(0, T-1)
    noise = torch.randn(images.shape, generator=generator, device=device)                              # eps ~ N(0, I)
    x_t = scheduler.add_noise(images, t, noise)           # sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    return F.mse_loss(model(x_t, t, tokens), noise)       # || eps - eps_theta(x_t, t, text) ||^2


def update_ema(ema, model, step, config):
    """Move every EMA weight a little towards the weight of the model: ema = decay * ema + (1 - decay) * weight.

    The decay grows from 0.1 to EMA_DECAY in the first steps: with 0.999 from the start, for the first
    ~1000 steps the EMA would still be mostly the random initial weights.

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


def main(config):
    """Train, validate, save a control grid every SAMPLE_EVERY epochs and the checkpoint at every epoch.

    Args:
        config: the project Config (uses the Training section and the settings of the model).
    """
    print('Training of the cartoon diffusion model')
    torch.manual_seed(config.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. One folder per experiment: the runs never overwrite each other
    name = 'conditional' if config.TEXT_CONDITIONING else 'unconditional'
    if config.TRAIN_SUBSET:
        name += f'_subset{config.TRAIN_SUBSET}'
    run_dir = config.RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'1. Experiment: {run_dir} (device: {device})')

    # 2. The data prepared by prepare_data.py
    train = CartoonDataset(config, 'train')
    vocabulary = train.vocabulary
    if config.TRAIN_SUBSET:
        train = Subset(train, range(config.TRAIN_SUBSET))    # smoke test: only the first images
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

    # 4. Resume from the checkpoint of the last epoch, if there is one
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

    # the settings of this run, as plain values (paths become text): the checkpoint opens on any computer
    settings = json.loads(json.dumps({name: getattr(config, name) for name in dir(config) if name.isupper()},
                                     default=str))
    # the captions of the control grid, twice: 2 images per caption
    grid_tokens = torch.tensor([vocabulary.encode(prompt) for prompt in config.SAMPLE_PROMPTS] * 2, device=device)
    step = (first_epoch - 1) * len(train_loader)     # training steps already done (for the EMA)

    print(f'5. Epochs {first_epoch}-{config.EPOCHS}')
    for epoch in range(first_epoch, config.EPOCHS + 1):
        start = time.time()
        model.train()
        total = 0.0
        for images, tokens in train_loader:
            images, tokens = images.to(device), tokens.to(device)
            if config.TEXT_CONDITIONING:
                # classifier-free guidance: a share of the captions becomes the empty caption
                drop = torch.rand(len(tokens), device=device) < config.CAPTION_DROPOUT
                tokens = torch.where(drop[:, None], model.empty_tokens, tokens)
            loss = diffusion_loss(model, scheduler, images, tokens)
            optimizer.zero_grad()
            loss.backward()                                                      # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()                                                     # AdamW updates the weights
            update_ema(ema, model, step, config)
            step += 1
            total += loss.item()
        train_loss = total / len(train_loader)

        # validation: EMA weights, real captions, the same t and noise at every epoch (fixed seed)
        generator = torch.Generator(device=device).manual_seed(config.SEED)
        with torch.no_grad():
            val_loss = sum(diffusion_loss(ema, scheduler, images.to(device), tokens.to(device), generator).item()
                           for images, tokens in val_loader) / len(val_loader)
        seconds = time.time() - start

        log_file = run_dir / 'log.csv'
        new_log = not log_file.exists()
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_log:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'seconds'])
            writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{seconds:.1f}'])
        print(f'   epoch {epoch}/{config.EPOCHS}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, {seconds:.0f} s')

        if config.SAMPLE_EVERY and epoch % config.SAMPLE_EVERY == 0:
            # control grid: same captions and seed every time, only the model changes
            images = sample(ema, scheduler, grid_tokens, config, config.SEED)
            grid_file = run_dir / f'samples_epoch{epoch:03d}.png'
            ImagePreprocessor(config).save_preview([ImagePreprocessor.denormalize(image) for image in images], grid_file)
            print(f'   control grid: {grid_file.name}')

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
            'settings': settings,
        }, checkpoint_file)

    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main(Config())
