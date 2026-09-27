# Avatar_Diffusion_Model_DL

Tiny text-conditioned diffusion model for cartoon avatars, trained from scratch on the
Google Cartoon Set (Deep Learning 2026_VI).

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # first: PyTorch with CUDA (NVIDIA GPU)
pip install -r requirements.txt
```

The first line matters on Windows: a plain `pip install torch` installs a CPU-only PyTorch, and the training
would take days. If PyTorch is already installed without CUDA, add `--force-reinstall` to the first line. Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

must print a version ending in `+cu126` and `True`. With more than one Python on the computer, make sure the
terminal (and the interpreter selected in VS Code) is the one with this PyTorch. On Colab PyTorch already has CUDA.

## Data

The raw data is not in the repository. Put folder 0 of cartoonset100k here:

```
dataset/cartoonset100k/0/0_images/      # the PNG files
dataset/cartoonset100k/0/0_attributes/  # the CSV files
```

Then prepare the data for training:

```bash
python prepare_data.py
```

All the settings are in `config.py`. The results go to `data/` (not in the repository):

| File | Content |
|---|---|
| `captions.csv` | all the images: words, original values, caption |
| `captions_train.csv`, `captions_val.csv`, `captions_test.csv` | in-distribution split (test = ordinary test) |
| `captions_ood.csv` | held-out combinations, column `ood_pair` says which one |
| `images_32/` | the images at 32x32 on a white background (values 0-255); `images_64/` with `IMAGE_SIZE = 64` |
| `vocabulary.json` | vocabulary built from the train captions only |
| `config.json` | the settings used, for the report |
| `summary.json` | numbers for the report: split sizes, held-out pairs, attributes in train |
| `preview.png` | the first 64 images |

**Images at 64x64.** Set `IMAGE_SIZE = 64` in `config.py` (in the file, not on a `Config` object in a notebook:
the images folder is chosen when `config.py` is read) and run `python prepare_data.py` again: it writes
`data/images_64/` next to `images_32/`, with the same captions, split and vocabulary (`preview.png`,
`config.json` and `summary.json` become the 64x64 ones). `python train.py` then trains the 64x64 UNet, which
has one more level (64, 32, 16, 8), in `runs/conditional_64/`: about 4 hours with batch 128 (6.4 GB) on an
RTX 4060 Laptop. Set `IMAGE_SIZE = 32` to go back.

## Code

| File | What it does |
|---|---|
| `config.py` | all the settings |
| `prepare_data.py` | runs the data pipeline in order |
| `preprocessing/caption_generator.py` | `CaptionGenerator`: attribute files -> words -> caption |
| `preprocessing/dataset_splitter.py` | `DatasetSplitter`: OOD pairs, then test/val/train by combination, checks |
| `preprocessing/vocabulary.py` | `Vocabulary`: tokens <-> ids, built from train only |
| `preprocessing/image_preprocessor.py` | `ImagePreprocessor`: white background, 32x32, normalization to [-1, 1] |
| `preprocessing/cartoon_dataset.py` | `CartoonDataset`: PyTorch Dataset of one split |
| `models/sinusoidal.py` | `sinusoidal_embedding`: sine/cosine vectors for token positions (and the UNet time steps) |
| `models/attention.py` | `MultiHeadAttention` written from scratch; `build_attention` gives it or `nn.MultiheadAttention` (`Config.ATTENTION`) |
| `models/text_encoder.py` | `EncoderBlock` and `TextEncoder`: caption ids -> one vector per token + `<pad>` mask |
| `models/noise_scheduler.py` | `NoiseScheduler`: cosine noise schedule (`NUM_TIMESTEPS` steps); `add_noise` gives the noisy images x_t of steps t |
| `models/unet.py` | `ResBlock`, `CrossAttention` (the pixels read the caption), `UNetBlock`, `UNet`: predicts the noise of x_t; `TEXT_CONDITIONING = False` gives the unconditional baseline; `IMAGE_SIZE = 64` adds a level outside, so the bottom is still 8x8 |
| `models/diffusion.py` | `DiffusionModel`: text encoder + UNet, from noisy images and captions to the predicted noise |
| `models/sampling.py` | `sample`: from pure noise to images (DDPM Algorithm 2), classifier-free guidance, seed |
| `train.py` | the training (DDPM Algorithm 1): loss, EMA, validation, control grids, checkpoint and resume |
| `make_training_gif.py` | the control grids of a training in one GIF, with the epoch written on top |
| `generate.py` | load a trained experiment (settings and EMA weights from its checkpoint) and generate one image per seed |
| `app.py` | web demo (Gradio): attributes from menus, seed, number of images, guidance, experiment; shows and saves the images |

## Use in training

```python
from torch.utils.data import DataLoader
from config import Config
from models.text_encoder import TextEncoder
from preprocessing.cartoon_dataset import CartoonDataset

train = CartoonDataset(Config(), 'train')   # images in [-1, 1], caption ids
loader = DataLoader(train, batch_size=128, shuffle=True)
text_encoder = TextEncoder(Config(), train.vocabulary)

for images, tokens in loader:
    context, pad_mask = text_encoder(tokens)   # (B, 18, D_MODEL), (B, 18): the condition of the UNet
```

## Training

After `prepare_data.py`:

```bash
python train.py                  # the real training
python train.py --smoke-test     # the smoke test
```

The settings are in the `# ---- Training ----` and `# ---- Smoke test ----` sections of `config.py`. Every
experiment writes to its own folder in `runs/` (not in the repository):

| Run | Folder |
|---|---|
| `python train.py`, `TEXT_CONDITIONING = True` | `runs/conditional_32/` (`_64` with `IMAGE_SIZE = 64`) |
| `python train.py`, `TEXT_CONDITIONING = False` | `runs/unconditional_32/` (baseline) |
| `python train.py --smoke-test` | `runs/conditional_smoke_test_32/` |

| File | Content |
|---|---|
| `last.pt` | checkpoint: weights, EMA weights, optimizer, epoch, random state, settings |
| `log.csv` | one row every `CHECKPOINT_EVERY` epochs: mean train loss since the previous row, validation loss, seconds |
| `samples_epochXXX.png` | control grid every `SAMPLE_EVERY` epochs, EMA weights: the 8 `SAMPLE_PROMPTS` (4 seen, 4 held-out), 2 images each |

If `last.pt` exists, the same command resumes from the next epoch; delete the folder to start again. On Colab set
`RUNS_DIR` to a folder of Google Drive, so the checkpoints survive the end of the session.

To see how the training improves the images, put its control grids in one GIF (one frame per grid, the epoch
written on top, 0.5 s per frame):

```bash
python make_training_gif.py runs/conditional_32      # writes runs/conditional_32/training.gif
```

`DETERMINISTIC = True` (for both commands) makes the GPU always add up in the same order: two trainings from the
same seed give exactly the same model, about 30% slower. With `False` they give almost the same model (the GPU
changes the order of its sums). Generating from a saved checkpoint with the same seed gives the same images in
both cases.

**Smoke test** (`--smoke-test`): the model trains only on the first 64 training images, with their captions, and
must learn them by heart. 8000 epochs of one step each, validation and checkpoint every 100 epochs: about 20
minutes on an RTX 4060 Laptop. Every 1000 epochs a grid: the first row shows 8 real training images, the two rows
below show images generated from their captions, which must look more and more like the first row. The train loss
goes down a lot; the validation loss goes up after a while (the model memorizes, it does not generalize: expected).

## Demo

```bash
python app.py            # then open the address it prints, e.g. http://127.0.0.1:7860
```

A web page: choose the experiment (the folders of `runs/` with a checkpoint), the five attributes, the seed of
the first image, the number of images and the guidance, then press **Generate**. The prompt is composed from the
attributes with the training template, and a warning appears for the held-out combinations (OOD). Every image
has its own seed (the next ones get seed + 1, + 2, ...): the same prompt, seed and guidance always give the
same image. With the box checked, the images are saved in `runs/<experiment>/generated/`, named after the
attributes, the guidance and the seed. One image takes about 19 s at 32x32 on an RTX 4060 Laptop. On Colab,
`python app.py --share` also prints a public link.

## Tests

From the project folder (they do not need `data/`):

```bash
python -m pytest tests/
```

## Colab

```python
!git clone https://github.com/diegopepe26/Avatar_Diffusion_Model_DL.git
%cd Avatar_Diffusion_Model_DL
!pip install -r requirements.txt
```

Then upload your local `data/` folder into `Avatar_Diffusion_Model_DL/` (zip it first:
10,000 small PNG files upload slowly one by one).
