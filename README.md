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

## Use in training

```python
from torch.utils.data import DataLoader
from config import Config
from preprocessing.cartoon_dataset import CartoonDataset

train = CartoonDataset(Config(), 'train')   # images in [-1, 1], caption ids
loader = DataLoader(train, batch_size=128, shuffle=True)
```

## Colab

```python
!git clone https://github.com/diegopepe26/Avatar_Diffusion_Model_DL.git
%cd Avatar_Diffusion_Model_DL
!pip install -r requirements.txt
```

Then upload your local `data/` folder into `Avatar_Diffusion_Model_DL/` (zip it first:
10,000 small PNG files upload slowly one by one).
