# Avatar_Diffusion_Model_DL

Tiny text-conditioned diffusion model for cartoon avatars, trained from scratch on the
Google Cartoon Set (Deep Learning 2026_VI).

## Setup

```bash
pip install -r requirements.txt
```

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
| `images_32/` | the images at 32x32 on a white background (values 0-255) |
| `vocabulary.json` | vocabulary built from the train captions only |
| `config.json` | the settings used, for the report |
| `summary.json` | numbers for the report: split sizes, held-out pairs, attributes in train |
| `preview.png` | the first 64 images |

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
| `models/unet.py` | `ResBlock`, `CrossAttention` (the pixels read the caption), `UNetBlock`, `UNet`: predicts the noise of x_t; `TEXT_CONDITIONING = False` gives the unconditional baseline |
| `models/diffusion.py` | `DiffusionModel`: text encoder + UNet, from noisy images and captions to the predicted noise |
| `models/sampling.py` | `sample`: from pure noise to images (DDPM Algorithm 2), classifier-free guidance, seed |
| `train.py` | the training (DDPM Algorithm 1): loss, EMA, validation, control grids, checkpoint and resume |

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
python train.py
```

The settings are in the `# ---- Training ----` section of `config.py`. Every experiment writes to its own
folder in `runs/` (not in the repository):

| Settings | Folder |
|---|---|
| `TEXT_CONDITIONING = True` | `runs/conditional/` |
| `TEXT_CONDITIONING = False` | `runs/unconditional/` (baseline) |
| `TRAIN_SUBSET = 64` | `runs/conditional_subset64/` (smoke test) |

| File | Content |
|---|---|
| `last.pt` | checkpoint of the last epoch: weights, EMA weights, optimizer, epoch, random state, settings |
| `log.csv` | train and validation loss of every epoch |
| `samples_epochXXX.png` | control grid: the 8 `SAMPLE_PROMPTS` (4 seen, 4 held-out), 2 images each, EMA weights |

If `last.pt` exists, `python train.py` resumes from the next epoch; delete the folder to start again.
Smoke test: `TRAIN_SUBSET = 64` and a high `EPOCHS` (for example 2000): the loss must go down a lot and the
grid must look like the training images. On Colab set `RUNS_DIR` to a folder of Google Drive, so the
checkpoints survive the end of the session.

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
