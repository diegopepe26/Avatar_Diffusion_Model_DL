# Text encoder: design

Data: 2026-09-24 · Branch: `text-encoder`

## Obiettivo

Trasformare gli id di una caption (output di `Vocabulary.encode`, lunghezza fissa `max_length` = 18) in una
sequenza di vettori di dimensione `D_MODEL`, che la UNet userà come condizionamento tramite cross-attention.
Il text encoder è un piccolo Transformer encoder addestrato da zero insieme alla UNet (stessa loss, stesso
optimizer).

## Vincoli di stile

- Stesso stile dei file in `preprocessing/`: classi semplici, docstring con Args/Returns, commenti brevi in
  inglese che spiegano il *perché*, impostazioni lette da `Config`, `ValueError` con messaggio chiaro.
- Solo l'essenziale: ogni classe ha `__init__` e `forward`, nessun metodo o opzione non discussi qui.

## File

```
config.py                  + sezione "Text encoder"
requirements.txt           + pytest
models/__init__.py         solo una docstring, come preprocessing/__init__.py
models/sinusoidal.py       sinusoidal_embedding(positions, dim)
models/attention.py        MultiHeadAttention, build_attention(config)
models/text_encoder.py     EncoderBlock, TextEncoder
tests/test_models.py       test con pytest
```

- `sinusoidal.py` è separato perché la UNet lo riuserà per i time step embedding.
- `attention.py` è separato perché la UNet riuserà `MultiHeadAttention` per la cross-attention
  (query dall'immagine, key e value dal testo).
- `EncoderBlock` e `TextEncoder` stanno nello stesso file: il blocco è usato solo dal text encoder.

## Configurazione (`config.py`)

```python
# ---- Text encoder ----
D_MODEL = 128             # size of the embeddings, used by the whole model (64 or 128)
NUM_HEADS = 4             # D_MODEL must be divisible by NUM_HEADS
NUM_ENCODER_BLOCKS = 2    # 2 for now, 4 if needed
FFN_DIM = 4 * D_MODEL     # hidden size of the feed-forward (two linear layers)
DROPOUT = 0.1
ATTENTION = 'scratch'     # 'scratch' (our MultiHeadAttention) or 'torch' (nn.MultiheadAttention)
```

- `vocab_size`, `max_length` e l'id di `<pad>` non stanno in `Config`: il `TextEncoder` li legge dall'oggetto
  `Vocabulary`, così restano allineati a `data/vocabulary.json` (stesso schema di `CartoonDataset`).
- `to_dict()` non cambia: descrive le impostazioni dei dati. Le impostazioni del modello si salveranno con
  il checkpoint, nello script di training.
- `BATCH_SIZE` non riguarda il modello: si aggiungerà a `Config` insieme al training e lo userà solo il
  `DataLoader`. Tutti i moduli funzionano con qualsiasi B.

## Componenti

Notazione: B = batch, L = `max_length` (18), D = `D_MODEL`, H = `NUM_HEADS`, d_k = D / H.

### `sinusoidal_embedding(positions, dim)` (`models/sinusoidal.py`)

- Input: `positions` tensore `(N,)`: posizioni dei token `0..L-1` oppure time step `t`.
- Output: `(N, dim)`, colonne pari seno e dispari coseno:
  `PE(pos, 2i) = sin(pos / 10000^(2i/dim))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))`.
- Il risultato è creato sul device di `positions`.
- `ValueError` se `dim` è dispari.

### `MultiHeadAttention(d_model, num_heads, dropout)` (`models/attention.py`)

Implementazione da zero con la **stessa interfaccia di `nn.MultiheadAttention(..., batch_first=True)`**,
così le due sono intercambiabili.

- `__init__`: quattro `nn.Linear(d_model, d_model)` (q, k, v, out) e un `nn.Dropout`;
  `ValueError` se `d_model % num_heads != 0`.
- `forward(query, key, value, key_padding_mask=None)`:
  1. Q, K, V: `(B, L, D)` → divisi in head → `(B, H, L, d_k)`
  2. `scores = Q @ Kᵀ / √d_k` → `(B, H, L_q, L_k)`
  3. colonne dei `<pad>` a `-inf` (`key_padding_mask` `(B, L_k)`, `True` = da ignorare, come in PyTorch)
  4. `weights = softmax(scores)`, poi dropout sui pesi (come fa `nn.MultiheadAttention`)
  5. `out = weights @ V` → head riunite → `(B, L_q, D)` → Linear out
  6. `return out, weights.mean(dim=1)`: pesi mediati sulle head `(B, L_q, L_k)`, come PyTorch

### `build_attention(config)` (`models/attention.py`)

- `config.ATTENTION == 'scratch'` → `MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT)`
- `config.ATTENTION == 'torch'` → `nn.MultiheadAttention(D_MODEL, NUM_HEADS, dropout=DROPOUT, batch_first=True)`
- altro valore → `ValueError`

### `EncoderBlock(config)` (`models/text_encoder.py`)

Versione classica del Transformer: add & norm dopo ogni sotto-blocco (post-LN).

- `__init__`: `attention = build_attention(config)`, `norm1`, `norm2` (`nn.LayerNorm(D)`),
  `feed_forward = Linear(D, FFN_DIM) → ReLU → Linear(FFN_DIM, D)`, `dropout`.
- `forward(x, pad_mask)`, `x` `(B, L, D)`:
  ```
  attended, _ = attention(x, x, x, key_padding_mask=pad_mask)
  x = norm1(x + dropout(attended))           # add & norm
  x = norm2(x + dropout(feed_forward(x)))    # add & norm
  return x                                   # (B, L, D)
  ```

### `TextEncoder(config, vocabulary)` (`models/text_encoder.py`)

- `__init__`:
  - `embedding = nn.Embedding(len(vocabulary.tokens), D, padding_idx=<pad>)`
  - `positional = sinusoidal_embedding(arange(vocabulary.max_length), D)` salvato con `register_buffer`
    (non addestrabile, ma segue il modello su GPU e nel `state_dict`), forma `(L, D)`
  - `dropout`
  - `blocks = nn.ModuleList([EncoderBlock(config) for _ in range(NUM_ENCODER_BLOCKS)])`
    (`ModuleList` e non una lista Python, altrimenti i pesi dei blocchi non sono registrati;
    non `nn.Sequential`, perché ogni blocco deve ricevere anche `pad_mask`)
- `forward(tokens)`, `tokens` `(B, L)` interi:
  ```
  pad_mask = tokens == pad_id                   # (B, L)
  x = dropout(embedding(tokens) + positional)   # (B, L, D); (L, D) si somma a ogni caption
  for block in blocks: x = block(x, pad_mask)
  return x, pad_mask                            # (B, L, D), (B, L)
  ```
- Gli embedding **non** si moltiplicano per √D (a differenza del paper): `nn.Embedding` parte già con
  valori dell'ordine di ±1 come seno e coseno, e ×√128 ≈ 11 schiaccerebbe la posizione. Commento nel codice.

## Perché il positional encoding

Senza, la self-attention non vede l'ordine delle parole: "no glasses and a beard" e "glasses and no beard"
darebbero gli stessi vettori in ordine diverso, e il "no" non saprebbe a quale attributo riferirsi.

## Flusso dei dati verso la UNet (riferimento, non implementato qui)

```
DataLoader        → images (B, 3, 32, 32), tokens (B, 18)
TextEncoder       → context (B, 18, D), pad_mask (B, 18)
UNet(x_t, t, context, pad_mask) con cross-attention → rumore predetto (B, 3, 32, 32)
```

La caption i-esima condiziona l'immagine i-esima; in generazione B è il numero di immagini da produrre.

## Test (`tests/test_models.py`, pytest)

Nessun file da `data/`: il vocabolario si costruisce nel test con `Vocabulary.build` su alcune caption.

1. `sinusoidal_embedding`: forma `(N, dim)`; posizione 0 → seno 0 e coseno 1; `dim` dispari → `ValueError`.
2. Equivalenza: si copiano i pesi di `MultiHeadAttention` (q, k, v → `in_proj_weight`/`in_proj_bias`,
   out → `out_proj`) in `nn.MultiheadAttention`; in `eval()`, con la stessa maschera di padding, output e
   pesi coincidono (`torch.allclose`).
3. `TextEncoder`, con `ATTENTION = 'scratch'` e `'torch'`: output `(B, 18, D)`, `pad_mask` `True`
   esattamente dove il token è `<pad>`.

Esecuzione: `python -m pytest tests/`. I test si scrivono prima del codice (TDD).

## Fuori da questo lavoro

UNet, cross-attention nella UNet (servirà `kdim`/`vdim` se le dimensioni di testo e immagine sono diverse),
time step embedding, training loop, `BATCH_SIZE`, checkpoint.
