# Text Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un text encoder Transformer (embedding + positional encoding sinusoidale + blocchi di encoder) che trasforma gli id di una caption in `(B, 18, D_MODEL)` vettori più la maschera dei `<pad>`, pronti per la cross-attention della UNet.

**Architecture:** Nuovo package `models/` con tre file: `sinusoidal.py` (funzione riusabile anche per i time step della UNet), `attention.py` (multi-head attention scritta da zero con la stessa interfaccia di `nn.MultiheadAttention`, scelta con `Config.ATTENTION`), `text_encoder.py` (`EncoderBlock` post-LN e `TextEncoder`). Tutte le dimensioni vengono da una nuova sezione di `config.py`; vocab size, `max_length` e id di `<pad>` vengono dall'oggetto `Vocabulary`.

**Tech Stack:** Python, PyTorch 2.8, pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-24-text-encoder-design.md`

## Global Constraints

- Stile identico ai file di `preprocessing/`: docstring con `Args:`/`Returns:`, commenti brevi in inglese che spiegano il *perché*, impostazioni lette da `Config`, `ValueError` con messaggio chiaro.
- Solo l'essenziale: ogni classe ha solo `__init__` e `forward`; nessun metodo, parametro o opzione non presente in questo piano.
- Codice, docstring e commenti in inglese (come il resto del repository).
- `D_MODEL = 128`, `NUM_HEADS = 4`, `NUM_ENCODER_BLOCKS = 2`, `FFN_DIM = 4 * D_MODEL`, `DROPOUT = 0.1`, `ATTENTION = 'scratch'`.
- Maschera di padding: `True` = token `<pad>` da ignorare (convenzione di `nn.MultiheadAttention`).
- I test si lanciano **dalla cartella del progetto** con `python -m pytest tests/` (non con `pytest` da solo: serve la cartella del progetto in `sys.path` per importare `config`, `models`, `preprocessing`). I test non leggono nulla da `data/`.
- Branch: `text-encoder`. Ogni commit termina con la riga `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Modello spostato su GPU o salvato con `torch.save`**: il positional encoding deve seguirlo → è un buffer registrato, non un tensore qualsiasi (test in Task 3: `test_text_encoder_trains_every_weight`).
- **Pesi di tutti i blocchi addestrati**: con una lista Python invece di `nn.ModuleList` l'optimizer non li vedrebbe → ogni parametro riceve un gradiente (Task 3, stesso test).
- **Un solo prompt in generazione (B = 1)**: l'output deve essere `(1, 18, D_MODEL)` → Task 3, `test_text_encoder`.
- **Errore di battitura in `Config.ATTENTION`** (es. `'Scratch'`): deve fermarsi con `ValueError`, non scegliere in silenzio un'implementazione → Task 2, `test_build_attention`.
- **`D_MODEL` dispari o non divisibile per `NUM_HEADS`**: `ValueError` chiaro invece di un errore di reshape incomprensibile → Task 1, `test_sinusoidal_embedding_odd_dim`; Task 2, `test_attention_heads_must_divide_d_model`.

---

### Task 1: package `models/`, pytest e `sinusoidal_embedding`

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `models/__init__.py`
- Create: `models/sinusoidal.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: niente.
- Produces: `sinusoidal_embedding(positions, dim) -> Tensor (N, dim)`, in `models/sinusoidal.py`. `positions` è un tensore `(N,)` (interi o float); il risultato sta sul device di `positions`; `ValueError` se `dim` è dispari.

- [ ] **Step 1: aggiungere pytest e ignorare la sua cache**

`requirements.txt` diventa:

```
numpy
pandas
Pillow
pytest
torch
```

In `.gitignore`, nella sezione `# Python`, aggiungere `.pytest_cache/` dopo `*.pyc`:

```
# Python
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: scrivere i test che falliscono**

Creare `tests/test_models.py`:

```python
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
```

- [ ] **Step 3: verificare che falliscano**

Run: `python -m pytest tests/ -v`
Expected: errore di collection, `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 4: creare il package e la funzione**

`models/__init__.py` (una riga, come `preprocessing/__init__.py`):

```python
"""The networks of the diffusion model: text encoder (and later the UNet)."""
```

`models/sinusoidal.py`:

```python
"""Sinusoidal embedding: token positions for the text encoder, time steps for the UNet."""

import torch


def sinusoidal_embedding(positions, dim):
    """Turn each position into a vector of sines and cosines at different frequencies.

    PE(pos, 2i) = sin(pos / 10000^(2i/dim)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/dim)).
    Nothing to train: the same position always gives the same vector.

    Args:
        positions: tensor (N,), e.g. the token positions 0..17 or the time steps t of a batch.
        dim: size of each vector (even), e.g. D_MODEL.

    Returns:
        Tensor (N, dim), on the same device as positions.
    """
    if dim % 2 != 0:
        raise ValueError(f'dim must be even (one sine and one cosine per frequency), not {dim}')
    two_i = torch.arange(0, dim, 2, device=positions.device)      # 0, 2, 4, ..., dim - 2
    frequencies = 1 / 10000 ** (two_i / dim)                       # (dim/2,)
    angles = positions.float()[:, None] * frequencies[None, :]     # (N, dim/2)
    embedding = torch.zeros(len(positions), dim, device=positions.device)
    embedding[:, 0::2] = torch.sin(angles)   # even columns
    embedding[:, 1::2] = torch.cos(angles)   # odd columns
    return embedding
```

- [ ] **Step 5: verificare che passino**

Run: `python -m pytest tests/ -v`
Expected: `2 passed`

- [ ] **Step 6: commit**

```bash
git add requirements.txt .gitignore models/__init__.py models/sinusoidal.py tests/test_models.py
git commit -m "feat: sinusoidal embedding for token positions and time steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: impostazioni in `config.py`, `MultiHeadAttention` e `build_attention`

**Files:**
- Modify: `config.py` (docstring del modulo, riga 1; docstring della classe, riga 7; nuova sezione prima di `def to_dict`)
- Create: `models/attention.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: niente dai task precedenti.
- Produces:
  - `Config.D_MODEL`, `Config.NUM_HEADS`, `Config.NUM_ENCODER_BLOCKS`, `Config.FFN_DIM`, `Config.DROPOUT`, `Config.ATTENTION`.
  - `MultiHeadAttention(d_model, num_heads, dropout)`, con `forward(query, key, value, key_padding_mask=None) -> (output (B, L_q, D), weights (B, L_q, L_k))`; `ValueError` se `d_model % num_heads != 0`. Attributi `q`, `k`, `v`, `out` (`nn.Linear`).
  - `build_attention(config)`: restituisce `MultiHeadAttention` se `ATTENTION == 'scratch'`, `nn.MultiheadAttention(..., batch_first=True)` se `'torch'`, altrimenti `ValueError`. Le due si chiamano allo stesso modo: `attention(x, x, x, key_padding_mask=pad_mask)`.

- [ ] **Step 1: aggiungere le impostazioni a `config.py`**

Riga 1, da:

```python
"""All the settings of the project in one place: paths, captions, split, images, vocabulary."""
```

a:

```python
"""All the settings of the project in one place: paths, captions, split, images, vocabulary, text encoder."""
```

Docstring della classe, da:

```python
    """Settings of the data pipeline. Change them here, never inside the other files."""
```

a:

```python
    """Settings of the project. Change them here, never inside the other files."""
```

Subito dopo `SPECIAL_TOKENS = ...` (sezione `# ---- Vocabulary ----`) e prima di `def to_dict(self):`, aggiungere:

```python
    # ---- Text encoder ----
    D_MODEL = 128             # size of the embeddings, used by the whole model (64 or 128)
    NUM_HEADS = 4             # D_MODEL must be divisible by NUM_HEADS
    NUM_ENCODER_BLOCKS = 2    # 2 for now, 4 if needed
    FFN_DIM = 4 * D_MODEL     # hidden size of the feed-forward (two linear layers)
    DROPOUT = 0.1
    ATTENTION = 'scratch'     # 'scratch' (our MultiHeadAttention) or 'torch' (nn.MultiheadAttention)

```

`to_dict()` non cambia: descrive solo le impostazioni dei dati.

- [ ] **Step 2: scrivere i test che falliscono**

In `tests/test_models.py`, sostituire il blocco degli import con:

```python
import pytest
import torch
from torch import nn

from config import Config
from models.attention import MultiHeadAttention, build_attention
from models.sinusoidal import sinusoidal_embedding
```

e aggiungere in fondo al file:

```python


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    ours = MultiHeadAttention(d_model=16, num_heads=4, dropout=0.1)
    theirs = nn.MultiheadAttention(16, 4, dropout=0.1, batch_first=True)
    # same weights in both: PyTorch keeps q, k and v stacked in one matrix
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([ours.q.weight, ours.k.weight, ours.v.weight]))
        theirs.in_proj_bias.copy_(torch.cat([ours.q.bias, ours.k.bias, ours.v.bias]))
        theirs.out_proj.weight.copy_(ours.out.weight)
        theirs.out_proj.bias.copy_(ours.out.bias)
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


def test_attention_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=128, num_heads=3, dropout=0.1)


def test_build_attention():
    config = Config()
    config.ATTENTION = 'scratch'
    assert isinstance(build_attention(config), MultiHeadAttention)
    config.ATTENTION = 'torch'
    assert isinstance(build_attention(config), nn.MultiheadAttention)
    config.ATTENTION = 'Scratch'   # a typo must stop, not silently pick one of the two
    with pytest.raises(ValueError):
        build_attention(config)
```

- [ ] **Step 3: verificare che falliscano**

Run: `python -m pytest tests/ -v`
Expected: errore di collection, `ModuleNotFoundError: No module named 'models.attention'`

- [ ] **Step 4: scrivere `models/attention.py`**

```python
"""Multi-head attention written from scratch, interchangeable with nn.MultiheadAttention."""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with several heads.

    Same interface as nn.MultiheadAttention(..., batch_first=True): same arguments of forward,
    same outputs, same mask convention. So build_attention can give either of the two.
    """

    def __init__(self, d_model, num_heads, dropout):
        """Args:
            d_model: size of the input and output vectors (D_MODEL).
            num_heads: number of heads; each one works on d_model / num_heads values.
            dropout: probability of dropping an attention weight during training.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by num_heads ({num_heads})')
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """Every query vector collects the value vectors, weighted by how much its key matches.

        Args:
            query: (B, L_q, D). In self-attention query, key and value are the same tensor.
            key: (B, L_k, D).
            value: (B, L_k, D).
            key_padding_mask: (B, L_k) bool, True where the token is <pad> (ignored); None = no mask.

        Returns:
            (output (B, L_q, D), attention weights averaged over the heads (B, L_q, L_k)).
        """
        batch, query_length, d_model = query.shape
        key_length = key.shape[1]
        # (B, L, D) -> (B, L, heads, head_dim) -> (B, heads, L, head_dim): each head gets its own slice
        q = self.q(query).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).view(batch, key_length, self.num_heads, self.head_dim).transpose(1, 2)

        # how much each query matches each key; / sqrt(head_dim) keeps the softmax from saturating
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)    # (B, heads, L_q, L_k)
        if key_padding_mask is not None:
            # -inf becomes weight 0 after the softmax: nobody looks at the <pad> tokens
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        # dropout on the weights, where nn.MultiheadAttention puts it
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = weights @ v                                            # (B, heads, L_q, head_dim)

        # back to (B, L_q, D): the heads side by side
        output = output.transpose(1, 2).reshape(batch, query_length, d_model)
        return self.out(output), weights.mean(dim=1)


def build_attention(config):
    """Give the attention chosen in config.ATTENTION: ours or the one of PyTorch.

    Args:
        config: the project Config (uses ATTENTION, D_MODEL, NUM_HEADS, DROPOUT).

    Returns:
        MultiHeadAttention ('scratch') or nn.MultiheadAttention ('torch'), called the same way.
    """
    if config.ATTENTION == 'scratch':
        return MultiHeadAttention(config.D_MODEL, config.NUM_HEADS, config.DROPOUT)
    if config.ATTENTION == 'torch':
        # batch_first: inputs (B, L, D) like ours, not the default (L, B, D)
        return nn.MultiheadAttention(config.D_MODEL, config.NUM_HEADS, dropout=config.DROPOUT, batch_first=True)
    raise ValueError(f"ATTENTION must be 'scratch' or 'torch', not '{config.ATTENTION}'")
```

- [ ] **Step 5: verificare che passino**

Run: `python -m pytest tests/ -v`
Expected: `5 passed`

- [ ] **Step 6: commit**

```bash
git add config.py models/attention.py tests/test_models.py
git commit -m "feat: multi-head attention from scratch, swappable with nn.MultiheadAttention

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `EncoderBlock`, `TextEncoder` e README

**Files:**
- Create: `models/text_encoder.py`
- Modify: `tests/test_models.py`
- Modify: `README.md` (tabella "Code", sezione "Use in training", nuova sezione "Tests")

**Interfaces:**
- Consumes: `sinusoidal_embedding(positions, dim)` (Task 1); `build_attention(config)` e le impostazioni `D_MODEL`, `FFN_DIM`, `DROPOUT`, `NUM_ENCODER_BLOCKS` (Task 2); `Vocabulary` esistente (`ids`, `tokens`, `max_length`, `build`, `encode`).
- Produces:
  - `EncoderBlock(config)`, con `forward(x (B, L, D), pad_mask (B, L)) -> (B, L, D)`.
  - `TextEncoder(config, vocabulary)`, con `forward(tokens (B, L) long) -> (context (B, L, D), pad_mask (B, L) bool)`. Attributi: `embedding`, buffer `positional` `(L, D)`, `blocks` (`nn.ModuleList`), `pad_id`.

- [ ] **Step 1: scrivere i test che falliscono**

In `tests/test_models.py`, sostituire il blocco degli import con:

```python
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
```

e aggiungere in fondo al file:

```python


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
    assert len(encoder.blocks) == config.NUM_ENCODER_BLOCKS
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, f'{name} is not trained'
    # the positional encoding follows the model (GPU, saved weights) but is not trained
    assert 'positional' in dict(encoder.named_buffers())
    assert 'positional' not in dict(encoder.named_parameters())
```

Nota sulla loss del secondo test: non usare `context.sum()`. L'ultima `LayerNorm` rende nulla la somma lungo D, quindi i gradienti sarebbero tutti zero.

- [ ] **Step 2: verificare che falliscano**

Run: `python -m pytest tests/ -v`
Expected: errore di collection, `ModuleNotFoundError: No module named 'models.text_encoder'`

- [ ] **Step 3: scrivere `models/text_encoder.py`**

```python
"""Text encoder: from the ids of a caption to one vector per token, the condition of the UNet."""

import torch
from torch import nn

from models.attention import build_attention
from models.sinusoidal import sinusoidal_embedding


class EncoderBlock(nn.Module):
    """One Transformer encoder block: self-attention and feed-forward, each followed by add & norm."""

    def __init__(self, config):
        """Args:
            config: the project Config (uses D_MODEL, FFN_DIM, DROPOUT and the attention settings).
        """
        super().__init__()
        self.attention = build_attention(config)
        self.norm1 = nn.LayerNorm(config.D_MODEL)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.D_MODEL, config.FFN_DIM),
            nn.ReLU(),
            nn.Linear(config.FFN_DIM, config.D_MODEL),
        )
        self.norm2 = nn.LayerNorm(config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x, pad_mask):
        """Args:
            x: (B, L, D), one vector per token.
            pad_mask: (B, L) bool, True where the token is <pad>.

        Returns:
            (B, L, D): every vector now also carries information from the other tokens.
        """
        attended, _ = self.attention(x, x, x, key_padding_mask=pad_mask)   # self-attention: q, k, v all from x
        x = self.norm1(x + self.dropout(attended))                # add & norm
        x = self.norm2(x + self.dropout(self.feed_forward(x)))    # add & norm
        return x


class TextEncoder(nn.Module):
    """Embeddings + positional encoding + NUM_ENCODER_BLOCKS encoder blocks.

    Trained from scratch together with the UNet.
    """

    def __init__(self, config, vocabulary):
        """Args:
            config: the project Config (uses D_MODEL, DROPOUT, NUM_ENCODER_BLOCKS and the block settings).
            vocabulary: the loaded Vocabulary (gives the number of tokens, max_length and the <pad> id).
        """
        super().__init__()
        self.pad_id = vocabulary.ids['<pad>']
        # padding_idx: the <pad> vector stays zero and is never trained
        self.embedding = nn.Embedding(len(vocabulary.tokens), config.D_MODEL, padding_idx=self.pad_id)
        # buffer: not trained, but it follows the model to the GPU and into the saved weights
        positions = torch.arange(vocabulary.max_length)
        self.register_buffer('positional', sinusoidal_embedding(positions, config.D_MODEL))   # (L, D)
        self.dropout = nn.Dropout(config.DROPOUT)
        # ModuleList, not a Python list: otherwise the weights of the blocks would not be registered
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.NUM_ENCODER_BLOCKS)])

    def forward(self, tokens):
        """Args:
            tokens: (B, L) caption ids from Vocabulary.encode, L = max_length.

        Returns:
            (context (B, L, D) one vector per token, pad_mask (B, L) True where the token is <pad>).
        """
        pad_mask = tokens == self.pad_id
        # No * sqrt(D_MODEL) as in the paper: nn.Embedding starts around +-1 like sin and cos,
        # multiplying by sqrt(128) ~ 11 would drown the position.
        x = self.dropout(self.embedding(tokens) + self.positional)   # the same (L, D) added to every caption
        for block in self.blocks:
            x = block(x, pad_mask)
        return x, pad_mask
```

- [ ] **Step 4: verificare che passino**

Run: `python -m pytest tests/ -v`
Expected: `8 passed`

- [ ] **Step 5: aggiornare il README**

Nella tabella `## Code`, dopo la riga di `preprocessing/cartoon_dataset.py`, aggiungere:

```markdown
| `models/sinusoidal.py` | `sinusoidal_embedding`: sine/cosine vectors for token positions (and the UNet time steps) |
| `models/attention.py` | `MultiHeadAttention` written from scratch; `build_attention` gives it or `nn.MultiheadAttention` (`Config.ATTENTION`) |
| `models/text_encoder.py` | `EncoderBlock` and `TextEncoder`: caption ids -> one vector per token + `<pad>` mask |
```

Nella sezione `## Use in training`, sostituire il blocco di codice con:

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

Prima di `## Colab`, aggiungere:

````markdown
## Tests

From the project folder (they do not need `data/`):

```bash
python -m pytest tests/
```

````

- [ ] **Step 6: verifica finale e commit**

Run: `python -m pytest tests/ -v`
Expected: `8 passed`

```bash
git add models/text_encoder.py tests/test_models.py README.md
git commit -m "feat: text encoder with encoder blocks, embeddings and positional encoding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
