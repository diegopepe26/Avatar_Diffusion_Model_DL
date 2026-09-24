# UNet: design

Data: 2026-09-24 · Branch: `unet` (da `noise-scheduler`)
Diagramma: artifact "Avatar UNet" (https://claude.ai/artifact/4DoPEyy5jMG5ymm3BM7in9)

## Obiettivo

Il denoiser del DDPM: una UNet compatta, addestrata da zero, che riceve l'immagine rumorosa `x_t`, il passo
`t` e (se il condizionamento è attivo) il testo codificato dal `TextEncoder`, e **predice il rumore** aggiunto
da `NoiseScheduler.add_noise`. Il training, la loss, la CFG e il sampling non fanno parte di questo lavoro.

## Coerenza con la traccia

| Traccia | Scelta |
|---|---|
| base channels 64-128 | 64 (poi 128 e 256) |
| due o tre risoluzioni | tre: 32 → 16 → 8 |
| residual block con time embedding | ResBlock del DDPM, time embedding sommato dopo la prima convoluzione |
| condizionamento via cross-attention | cross-attention a 16 × 16 e 8 × 8 |
| U-Net forward, conditioning path scritti da noi | `models/unet.py`, attention da zero (`MultiHeadAttention`) |
| baseline non condizionata + modello condizionato | flag `TEXT_CONDITIONING` |
| parameter count, sampling time | misurati sul prototipo (vedi sotto) |

## Decisioni

- **Architettura "classica compatta"** (opzione A): ResBlock a ogni livello, cross-attention solo a 16 × 16 e
  8 × 8. A 32 × 32 le feature sono locali (bordi, sfumature); gli attributi (capelli, occhiali, barba)
  emergono a 16 × 16 e 8 × 8. Come nel DDPM, dove l'attention è a 16 × 16.
- **Due ResBlock per livello**, in discesa e in salita; **una skip per livello**, presa alla fine dei due
  ResBlock della discesa.
- **Fondo a 8 × 8: ResBlock → Cross-Attention → ResBlock**, come nei paper (DDPM, Stable Diffusion).
- **Downsampling con convoluzione 3 × 3 a stride 2** (non max pooling): impara come riassumere i pixel.
- **Upsampling con interpolazione nearest × 2 + convoluzione 3 × 3** (non convoluzione trasposta): evita gli
  artefatti a scacchiera.
- **Skip per concatenazione dei canali**, come nella U-Net originale e nel DDPM; il primo ResBlock dopo la
  concatenazione riporta i canali al valore del livello (384 → 128, 192 → 64).
- **Il testo entra direttamente in W_K e W_V**: `MultiHeadAttention` riceve `kdim` (numeri per parola, 128)
  diverso da `d_model` (canali del livello: 128 o 256). Niente Linear in più prima dell'attention: due
  trasformazioni lineari di fila equivalgono a una. `nn.MultiheadAttention` ha gli stessi argomenti
  (`kdim`, `vdim`), quindi lo scambio via `Config.ATTENTION` resta possibile anche nella UNet.
- **La UNet predice il rumore e basta**: la loss (MSE tra rumore predetto e vero) sta nel passo di training,
  così la stessa UNet si usa identica nel training e nella generazione.
- **Flag `TEXT_CONDITIONING`**: con `False` la UNet si costruisce senza cross-attention (baseline non
  condizionata richiesta dalla traccia) e si chiama senza `context`.

## Vincoli di stile

Come il resto del progetto: classi semplici con `__init__` e `forward`, `config` nel costruttore, docstring
con Args/Returns, commenti brevi in inglese sul *perché*, solo l'essenziale, `ValueError` con messaggio
chiaro.

## File

```
config.py                  + sezione "UNet"
models/attention.py        MultiHeadAttention(..., kdim=None); build_attention(config, d_model, kdim=None)
models/text_encoder.py     EncoderBlock: build_attention(config, config.D_MODEL)
models/unet.py             ResBlock, CrossAttention, UNetBlock, UNet
models/__init__.py         docstring senza "(and later the UNet)"
tests/test_models.py       chiamate a build_attention aggiornate + test di kdim
tests/test_unet.py         test della UNet
README.md                  una riga nella tabella "Code"
```

## Configurazione (`config.py`)

```python
# ---- UNet ----
UNET_CHANNELS = (64, 128, 256)   # channels at 32x32, 16x16, 8x8
TIME_DIM = 4 * UNET_CHANNELS[0]  # size of the time embedding given to every ResBlock
TEXT_CONDITIONING = True         # False: unconditional baseline, no cross-attention
```

Riusati: `D_MODEL` (numeri per parola del testo, `kdim` della cross-attention), `NUM_HEADS` (4 head anche
nella UNet: 128 / 4 = 32 e 256 / 4 = 64 numeri per head), `DROPOUT` (un solo valore per tutto il modello),
`ATTENTION`. Costante nel codice: GroupNorm a 32 gruppi, come nel DDPM (tutti i canali usati, 64, 128, 256,
192, 384, sono divisibili per 32).

## Componenti

### `models/attention.py` (modifica)

- `MultiHeadAttention(d_model, num_heads, dropout, kdim=None)`: `w_k` e `w_v` diventano
  `nn.Linear(kdim, d_model)`; con `kdim=None` valgono `d_model` (text encoder invariato).
  `forward` invariato: `x_k` e `x_v` hanno `kdim` numeri per token, `x_q` ne ha `d_model`.
- `build_attention(config, d_model, kdim=None)`: `d_model` = dimensione dell'attention (`D_MODEL` nel text
  encoder, canali del livello nella UNet); `'torch'` → `nn.MultiheadAttention(d_model, NUM_HEADS,
  dropout=DROPOUT, batch_first=True, kdim=kdim, vdim=kdim)`.

### `ResBlock(in_channels, out_channels, config)`

Come nel DDPM:

```
h = Conv3×3(SiLU(GroupNorm(x)))                       in → out
h = h + Linear(SiLU(time))  per canale                 time (B, TIME_DIM) → (B, out, 1, 1)
h = Conv3×3(Dropout(SiLU(GroupNorm(h))))               out → out
return shortcut(x) + h                                 shortcut = Conv1×1 se in ≠ out, altrimenti identità
```

### `CrossAttention(channels, config)`

```
pixels = GroupNorm(x) → (B, H·W, C)                    i pixel diventano token: x_q
attended = build_attention(config, C, kdim=D_MODEL)(pixels, context, context, key_padding_mask=pad_mask)
return x + attended → (B, C, H, W)                     residuo
```

`context` `(B, 18, D_MODEL)` e `pad_mask` `(B, 18)` sono l'uscita del `TextEncoder`.

### `UNetBlock(in_channels, out_channels, config, attention)`

Un `ResBlock` seguito da una `CrossAttention` se `attention` è vero **e** `TEXT_CONDITIONING` è vero
(altrimenti nessuna cross-attention). `forward(x, time, context, pad_mask)`. Serve a scrivere ogni livello
come una lista di blocchi uguali, con o senza testo.

### `UNet(config)`

Con `c1, c2, c3 = UNET_CHANNELS` (64, 128, 256):

```
time      = MLP(sinusoidal_embedding(t, c1))          Linear c1→TIME_DIM, SiLU, Linear TIME_DIM→TIME_DIM
conv_in     Conv3×3 3 → c1                                               32×32
down1       UNetBlock(c1, c1), UNetBlock(c1, c1)       senza testo       → skip1 (c1)
downsample1 Conv3×3 stride 2                                             32 → 16
down2       UNetBlock(c1, c2) + CA, UNetBlock(c2, c2) + CA               → skip2 (c2)
downsample2 Conv3×3 stride 2                                             16 → 8
middle      UNetBlock(c2, c3) + CA, UNetBlock(c3, c3)  = ResBlock → CA → ResBlock
upsample2   nearest ×2 + Conv3×3 (c3)                                    8 → 16
            cat(skip2) → c3 + c2 = 384
up2         UNetBlock(c3 + c2, c2) + CA, UNetBlock(c2, c2) + CA
upsample1   nearest ×2 + Conv3×3 (c2)                                    16 → 32
            cat(skip1) → c2 + c1 = 192
up1         UNetBlock(c2 + c1, c1), UNetBlock(c1, c1)  senza testo
conv_out    GroupNorm, SiLU, Conv3×3 c1 → 3            → rumore predetto
```

- `forward(x_t, t, context=None, pad_mask=None)` → `(B, 3, H, W)`, stessa forma di `x_t`.
- `x_t` `(B, 3, H, W)` con H e W divisibili per 4 (32 o 64); `t` `(B,)` interi.
- Con `TEXT_CONDITIONING = True` e `context=None`: `ValueError` con messaggio chiaro.
- Con `TEXT_CONDITIONING = False`: nessuna cross-attention; `context` e `pad_mask` ignorati.
- 5 cross-attention in tutto con il testo (2 in down2, 1 nel fondo, 2 in up2).

## Numeri misurati sul prototipo

- Parametri: **5.745.475** con testo, **5.282.115** senza (solo UNet; il `TextEncoder` ne ha altri 400.256).
- Tempo: 32,6 ms per chiamata della UNet con batch 64 su RTX 4060 Laptop → 1000 passi ≈ 33 s per 64
  immagini, ≈ 65 s con la CFG (due chiamate per passo). Sulla T4 di Colab sarà più lento.

## Uso previsto (riferimento, non implementato qui)

```
context, pad_mask = text_encoder(tokens)          # solo con TEXT_CONDITIONING = True
predetto = unet(x_t, t, context, pad_mask)
loss = MSE(predetto, noise)
```

CFG (decisa, da implementare con training e sampling): nel training circa il 10% delle caption si sostituisce
con la caption vuota `vocabulary.encode('')` (solo `<bos>`, `<eos>` e `<pad>`: una caption di soli `<pad>`
nasconderebbe tutte le parole e l'attention darebbe NaN); in generazione due chiamate per passo, con e senza
prompt, combinate con la guidance scale. La UNet non cambia.

## Test

`tests/test_models.py`:
- le chiamate a `build_attention(config)` diventano `build_attention(config, config.D_MODEL)`;
- nuovo: `MultiHeadAttention` con `kdim` diverso da `d_model` coincide con `nn.MultiheadAttention(kdim=...,
  vdim=...)` a pesi copiati (`q_proj_weight`, `k_proj_weight`, `v_proj_weight`, `in_proj_bias`, `out_proj`).

`tests/test_unet.py` (senza `data/`, input casuali):
1. Forma dell'uscita `(B, 3, 32, 32)`, con `ATTENTION = 'scratch'` e `'torch'`.
2. Immagini 64 × 64 → uscita `(B, 3, 64, 64)`.
3. `TEXT_CONDITIONING = False`: funziona senza `context`, nessuna `CrossAttention`; con `True` sono 5.
4. In `eval()`: un testo diverso cambia l'uscita, un `t` diverso cambia l'uscita, cambiare i vettori nelle
   posizioni `<pad>` non la cambia.
5. `TEXT_CONDITIONING = True` senza `context` → `ValueError`.
6. Ogni parametro riceve un gradiente (nessun blocco dimenticato fuori da una `ModuleList`).

Esecuzione: `python -m pytest tests/`. I test si scrivono prima del codice (TDD).

## Fuori da questo lavoro

`train.py` (loss, optimizer, checkpoint per riprendere, seed), CFG, reverse sampling loop, valutazione
(FID/KID, diversità tra seed, memoria), demo con prompt e seed.
