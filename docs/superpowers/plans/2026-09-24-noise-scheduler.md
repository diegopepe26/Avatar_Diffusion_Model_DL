# Noise Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Una classe `NoiseScheduler` che calcola la curva cosine del rumore (T = 1000 passi) e porta in un colpo solo un batch di immagini pulite al livello di rumore del loro passo `t` (forward diffusion), pronta per il ciclo di training della UNet.

**Architecture:** Nuovo file `models/noise_scheduler.py` con una classe semplice (non `nn.Module`): l'`__init__(config)` calcola una volta `betas` e `alpha_bars` (tensori `(T,)` sulla CPU); `add_noise(images, t, noise)` applica `x_t = √ᾱ_t · images + √(1 − ᾱ_t) · noise`. `T` viene da `Config.NUM_TIMESTEPS`. `CartoonDataset` non cambia.

**Tech Stack:** Python, PyTorch 2.8, pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-24-noise-scheduler-design.md`

## Global Constraints

- Stile identico a `preprocessing/` e `models/`: docstring con `Args:`/`Returns:`, commenti brevi in inglese sul *perché*, impostazioni lette da `Config`.
- Solo l'essenziale: `NoiseScheduler` ha solo `__init__` e `add_noise`; attributi `num_timesteps`, `betas`, `alpha_bars`. Nessun altro metodo o opzione.
- Codice, docstring e commenti in inglese.
- `NUM_TIMESTEPS = 1000`; costanti cosine nel codice: offset `0.008`, beta massimo `0.999`.
- `t` va da 0 a `NUM_TIMESTEPS − 1`; `t` e `noise` si passano da fuori, lo scheduler non li genera.
- I test si lanciano dalla cartella del progetto con `python -m pytest tests/`; non leggono nulla da `data/`.
- Branch: `noise-scheduler`. Ogni commit termina con la riga `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **`t` su GPU con la tabella sulla CPU**: `add_noise` deve funzionare senza spostare lo scheduler → Task 2, `test_add_noise_on_gpu` (saltato se non c'è CUDA).
- **`t` creato sulla CPU (es. `torch.randint` senza `device=`) con immagini su GPU**: deve funzionare lo stesso → Task 2, stesso test.
- **Immagini 64×64** (`Config.IMAGE_SIZE` può valere 64): `add_noise` non deve dipendere da 32 → Task 2, `test_add_noise_one_step_per_image` usa 64×64.
- **`t = 1000`** (errore "da 1 a T" invece di "da 0 a T−1"): deve dare errore, non un risultato sbagliato in silenzio → Task 2, `test_add_noise_step_out_of_range`.
- **Precisione del prodotto di 1000 fattori**: `alpha_bars` deve restare strettamente decrescente e l'ultimo valore quasi 0 → Task 1, `test_cosine_curve`.

---

### Task 1: `NUM_TIMESTEPS` in `config.py` e curva cosine

**Files:**
- Modify: `config.py` (docstring del modulo, riga 1; nuova sezione dopo `ATTENTION = ...`)
- Modify: `models/__init__.py`
- Create: `models/noise_scheduler.py`
- Create: `tests/test_noise_scheduler.py`

**Interfaces:**
- Consumes: niente.
- Produces: `Config.NUM_TIMESTEPS` (int, 1000); `NoiseScheduler(config)` con attributi `num_timesteps` (int), `betas` (float32 `(T,)`, CPU), `alpha_bars` (float32 `(T,)`, CPU).

- [ ] **Step 1: scrivere il test che fallisce**

Creare `tests/test_noise_scheduler.py`:

```python
"""Tests of the noise scheduler. Run them with:  python -m pytest tests/"""

import pytest
import torch

from config import Config
from models.noise_scheduler import NoiseScheduler


def test_cosine_curve():
    scheduler = NoiseScheduler(Config())
    assert scheduler.betas.shape == (1000,)
    assert scheduler.alpha_bars.shape == (1000,)
    assert scheduler.alpha_bars[0] > 0.999                                   # step 0: almost the clean image
    assert torch.all(scheduler.alpha_bars[1:] < scheduler.alpha_bars[:-1])   # always less image left
    assert abs(scheduler.alpha_bars[499] - 0.49) < 0.01                      # cosine: about half at the middle
    assert scheduler.alpha_bars[999] < 0.001                                 # last step: pure noise
    assert torch.all(scheduler.betas > 0) and torch.all(scheduler.betas <= 0.999)
```

(`pytest` si usa nei test del Task 2; l'import resta qui fin da subito.)

- [ ] **Step 2: verificare che fallisca**

Run: `python -m pytest tests/ -v`
Expected: errore di collection, `ModuleNotFoundError: No module named 'models.noise_scheduler'`

- [ ] **Step 3: aggiungere `NUM_TIMESTEPS` a `config.py`**

Riga 1, da:

```python
"""All the settings of the project in one place: paths, captions, split, images, vocabulary, text encoder."""
```

a:

```python
"""All the settings of the project in one place: paths, captions, split, images, vocabulary, models."""
```

Subito dopo la riga `ATTENTION = 'scratch'     # ...` (ultima della sezione `# ---- Text encoder ----`), prima della riga vuota e di `def to_dict(self):`, aggiungere:

```python

    # ---- Diffusion ----
    NUM_TIMESTEPS = 1000      # T: noise steps, t = 0 (almost clean) ... T - 1 (pure noise)
```

- [ ] **Step 4: aggiornare `models/__init__.py`**

Il file diventa:

```python
"""The parts of the diffusion model: text encoder, noise scheduler (and later the UNet)."""
```

- [ ] **Step 5: scrivere `models/noise_scheduler.py` (solo `__init__`)**

```python
"""Forward diffusion: the fixed recipe that brings a clean image to the noise level of step t."""

import math

import torch


class NoiseScheduler:
    """Cosine noise schedule (Nichol & Dhariwal 2021, Improved DDPM) and the noising of the images.

    Nothing to train: the same step t always gives the same amount of noise.
    """

    def __init__(self, config):
        """Compute, once, the noise of every step and how much of the image is left after it.

        Args:
            config: the project Config (uses NUM_TIMESTEPS).
        """
        self.num_timesteps = config.NUM_TIMESTEPS
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float64) / self.num_timesteps   # 0, 1/T, ..., 1
        # cosine curve of the paper; the 0.008 offset keeps the first steps from being too small
        curve = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        # noise added by the single step t, at most 0.999: the last step would otherwise be 1
        # (image completely gone) and the generation would divide by zero
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        self.betas = betas.float()                                    # (T,)
        # share of the image left after step t: product of (1 - beta) from step 0 to t
        # (computed in float64: it multiplies 1000 numbers)
        self.alpha_bars = torch.cumprod(1 - betas, dim=0).float()     # (T,)
```

- [ ] **Step 6: verificare che passi**

Run: `python -m pytest tests/ -v`
Expected: `9 passed` (8 del text encoder + `test_cosine_curve`)

- [ ] **Step 7: commit**

```bash
git add config.py models/__init__.py models/noise_scheduler.py tests/test_noise_scheduler.py
git commit -m "feat: cosine noise schedule with NUM_TIMESTEPS in config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `add_noise` e README

**Files:**
- Modify: `models/noise_scheduler.py` (nuovo metodo in fondo alla classe)
- Modify: `tests/test_noise_scheduler.py`
- Modify: `README.md` (tabella "Code")

**Interfaces:**
- Consumes: `NoiseScheduler(config)` con `alpha_bars` float32 `(T,)` sulla CPU (Task 1).
- Produces: `NoiseScheduler.add_noise(images (B, 3, H, W), t (B,) long, noise (B, 3, H, W)) -> x_t (B, 3, H, W)` sul device di `images`; `IndexError` se un `t` è fuori da `0 … T−1` (su CPU).

- [ ] **Step 1: scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_noise_scheduler.py`:

```python


def test_add_noise_extremes():
    scheduler = NoiseScheduler(Config())
    images = torch.rand(4, 3, 32, 32) * 2 - 1      # values in [-1, 1], like CartoonDataset
    noise = torch.randn(4, 3, 32, 32)
    first = scheduler.add_noise(images, torch.zeros(4, dtype=torch.long), noise)
    last = scheduler.add_noise(images, torch.full((4,), 999), noise)
    assert torch.allclose(first, images, atol=0.05)   # t = 0: almost the clean image
    assert torch.allclose(last, noise, atol=0.05)     # t = 999: almost only noise


def test_add_noise_one_step_per_image():
    scheduler = NoiseScheduler(Config())
    images = torch.rand(3, 3, 64, 64) * 2 - 1      # 64x64: IMAGE_SIZE can also be 64
    noise = torch.randn(3, 3, 64, 64)
    t = torch.tensor([0, 500, 999])
    noisy = scheduler.add_noise(images, t, noise)
    assert noisy.shape == images.shape
    for i in range(3):   # every image with the alpha_bar of its own step
        alpha_bar = scheduler.alpha_bars[t[i]]
        expected = alpha_bar.sqrt() * images[i] + (1 - alpha_bar).sqrt() * noise[i]
        assert torch.allclose(noisy[i], expected)


def test_add_noise_step_out_of_range():
    scheduler = NoiseScheduler(Config())
    images = torch.zeros(1, 3, 32, 32)
    with pytest.raises(IndexError):   # t goes from 0 to 999: 1000 does not exist
        scheduler.add_noise(images, torch.tensor([1000]), torch.randn(1, 3, 32, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_add_noise_on_gpu():
    scheduler = NoiseScheduler(Config())            # the table stays on the CPU
    images = torch.rand(2, 3, 32, 32, device='cuda') * 2 - 1
    noise = torch.randn(2, 3, 32, 32, device='cuda')
    t_gpu = torch.randint(0, 1000, (2,), device='cuda')
    assert scheduler.add_noise(images, t_gpu, noise).device.type == 'cuda'
    t_cpu = torch.randint(0, 1000, (2,))            # t made without device=...: must work too
    assert scheduler.add_noise(images, t_cpu, noise).device.type == 'cuda'
```

- [ ] **Step 2: verificare che falliscano**

Run: `python -m pytest tests/ -v`
Expected: `4 failed, 9 passed` con `AttributeError: 'NoiseScheduler' object has no attribute 'add_noise'` (senza GPU: `3 failed, 9 passed, 1 skipped`)

- [ ] **Step 3: aggiungere `add_noise` in fondo alla classe `NoiseScheduler`**

```python

    def add_noise(self, images, t, noise):
        """Bring every image to the noise level of its step t, in one go.

        The t small noise additions of the chain add up to a single one:
        x_t = sqrt(alpha_bar_t) * image + sqrt(1 - alpha_bar_t) * noise.

        Args:
            images: (B, 3, H, W) clean images in [-1, 1].
            t: (B,) integer steps between 0 and NUM_TIMESTEPS - 1, one per image.
            noise: (B, 3, H, W) Gaussian noise, passed from outside: the training needs it for the loss.

        Returns:
            (B, 3, H, W) noisy images x_t, on the device of images.
        """
        # the table lives on the CPU: move it to the device of the images, take one value per image
        # and shape it (B, 1, 1, 1), so it applies to all the channels and pixels of that image
        alpha_bar = self.alpha_bars.to(images.device)[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise
```

- [ ] **Step 4: verificare che passino**

Run: `python -m pytest tests/ -v`
Expected: `13 passed` (senza GPU: `12 passed, 1 skipped`)

- [ ] **Step 5: aggiornare il README**

Nella tabella `## Code`, dopo la riga di `models/text_encoder.py`, aggiungere:

```markdown
| `models/noise_scheduler.py` | `NoiseScheduler`: cosine noise schedule (`NUM_TIMESTEPS` steps); `add_noise` gives the noisy images x_t of steps t |
```

- [ ] **Step 6: verifica finale e commit**

Run: `python -m pytest tests/ -v`
Expected: `13 passed` (senza GPU: `12 passed, 1 skipped`)

```bash
git add models/noise_scheduler.py tests/test_noise_scheduler.py README.md
git commit -m "feat: add_noise brings the images to the noise level of step t

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
