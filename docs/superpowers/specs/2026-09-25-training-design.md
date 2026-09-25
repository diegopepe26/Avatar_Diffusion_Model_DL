# Training e sampling: design

Data: 2026-09-25 · Branch: `training` (da `unet`)

## Obiettivo

Addestrare il modello (text encoder + UNet) con l'algoritmo di training del DDPM e generare immagini dal
rumore con il reverse sampling loop del DDPM, con classifier-free guidance, EMA dei pesi, seed fisso e
checkpoint per riprendere. Serve per i due esperimenti richiesti dalla traccia (modello condizionato e
baseline non condizionata) e per lo smoke test consigliato. Valutazione (FID/KID, metriche di
condizionamento, diversità), `generate.py` e demo non fanno parte di questo lavoro.

## Coerenza con la traccia

| Traccia | Scelta |
|---|---|
| "sampling a timestep, adding noise, predicting the noise target" | `diffusion_loss` in `train.py` (algoritmo 1 del DDPM) |
| diffusion objective e reverse sampling loop scritti da noi | `train.py`, `NoiseScheduler.step`, `sample` |
| seed fisso per il run principale | `Config.SEED = 42` |
| "save enough information to resume training" | `last.pt` a ogni epoca, ripresa automatica |
| checkpointing | `last.pt` con pesi, EMA, optimizer, epoca, stato del generatore casuale, impostazioni |
| smoke test 32 × 32 che impara un piccolo sottoinsieme | `TRAIN_SUBSET = 64` |
| baseline non condizionata + modello condizionato | `TEXT_CONDITIONING`, una cartella per esperimento |
| budget T4 | stesso codice in locale (RTX 4060) e su Colab: cambia solo `RUNS_DIR` |

## Decisioni

- **Training prima in locale** sulla RTX 4060 (misurato: 226 ms per passo con batch 128, 10 s per epoca,
  2,4 GB di memoria), con il codice pronto per Colab. Niente mixed precision per ora: si aggiunge solo se
  lo smoke test mostra che serve.
- **`DiffusionModel` = solo le parti che imparano** (text encoder + UNet): `forward(x_t, t, tokens)` →
  rumore predetto. Un solo oggetto da spostare su GPU, copiare per l'EMA e salvare.
- **La loss sta in `train.py`**, in una funzione usata sia dal training sia dalla validation: si legge
  l'algoritmo 1 del DDPM riga per riga. La backpropagation (`loss.backward()`) e l'aggiornamento
  (`optimizer.step()`) stanno nel ciclo di `train.py`.
- **Il `NoiseScheduler` resta separato** (ricetta fissa, senza pesi) e riceve `step()`, il passo
  all'indietro della generazione, perché usa le stesse tabelle di `add_noise`.
- **Il passo all'indietro** stima l'immagine pulita rovesciando `add_noise`, la limita a [−1, 1] (come il
  codice ufficiale del DDPM: a `t = 999` √ᾱ ≈ 0,00005 moltiplicherebbe ogni errore per 20.000), poi fa un
  passo verso di lei con la media della posterior q(x_{t−1} | x_t, x_0) e aggiunge rumore nuovo con
  varianza β̃_t ("fixed small" del DDPM); a `t = 0` niente rumore nuovo. `t − 1` a `t = 0` è gestito
  esplicitamente (niente indice −1).
- **Classifier-free guidance**: nel training il 10% delle caption diventa la caption vuota
  `vocabulary.encode('')` (`<bos>`, `<eos>`, `<pad>`…: con soli `<pad>` l'attention darebbe NaN); in
  generazione due chiamate per passo, combinate con
  `ε̂ = ε̂_vuota + GUIDANCE_SCALE · (ε̂_prompt − ε̂_vuota)`. La baseline non condizionata fa una chiamata sola.
- **EMA dei pesi** con 0,999, e un avvio graduale: `decay = min(0,999, (1 + passo) / (10 + passo))`.
  Senza l'avvio, nei primi ~1000 passi (e in tutto lo smoke test) i pesi EMA sarebbero ancora vicini a
  quelli casuali iniziali. Si genera e si valida con i pesi EMA.
- **Seed**: `torch.manual_seed(SEED)` all'inizio del training; in generazione un `torch.Generator` con il
  seed dato: stesso prompt + stesso seed → stessa immagine.
- **Validation** a ogni epoca sull'intero split `val`, con i pesi EMA, senza caption vuote, con `t` e
  rumore estratti da un generatore con seed fisso: le loss delle diverse epoche sono confrontabili.
  Test e OOD non si usano nel training.
- **Griglia di controllo** ogni `SAMPLE_EVERY` epoche: le 8 caption fisse ripetute 2 volte = 16 avatar,
  generati con i pesi EMA in una sola chiamata di `sample` con seed `SEED`, salvati come PNG (8 colonne =
  caption, 2 righe = due immagini diverse per caption). Stesse caption e stesso seed a ogni griglia: le
  differenze dipendono solo da quanto ha imparato il modello.

## Vincoli di stile

Come il resto del progetto: classi semplici (`__init__` e `forward`, più il nuovo `step` dello scheduler,
che è una formula della stessa ricetta), funzioni brevi, docstring con Args/Returns, commenti brevi in
inglese sul *perché*, impostazioni in `config.py`, solo l'essenziale. `train.py` ha la forma di
`prepare_data.py`: una `main()` che esegue i passi in ordine e stampa cosa fa.

## File

```
config.py                              + sezione "Training"
models/noise_scheduler.py              + step(x_t, t, predicted_noise, generator=None)
models/diffusion.py                    DiffusionModel
models/sampling.py                     sample(model, scheduler, tokens, config, seed)
preprocessing/image_preprocessor.py    save_preview(images, path): percorso come parametro, righe secondo il numero di immagini
train.py                               diffusion_loss, update_ema, main(config)
tests/test_training.py                 test di step, DiffusionModel, sample, diffusion_loss, update_ema, save_preview
README.md                              sezione "Training"
.gitignore                             runs/
```

## Configurazione (`config.py`)

```python
# ---- Training ----
RUNS_DIR = ROOT / 'runs'          # one folder per experiment; on Colab point it to Google Drive
BATCH_SIZE = 128
LEARNING_RATE = 2e-4              # AdamW, constant, as in DDPM
GRAD_CLIP = 1.0                   # largest norm of the gradient, as in DDPM
EMA_DECAY = 0.999                 # average of the weights over the last ~1000 steps
CAPTION_DROPOUT = 0.1             # share of empty captions, for classifier-free guidance
EPOCHS = 500                      # ~22,000 steps, ~1.4 h on an RTX 4060 Laptop
SAMPLE_EVERY = 25                 # epochs between two control grids (0 = never)
TRAIN_SUBSET = 0                  # 0 = all the training images; 64 for the smoke test
GUIDANCE_SCALE = 3.0              # strength of classifier-free guidance when generating
SAMPLE_PROMPTS = [                # control grid: 4 captions seen in training, 4 held-out (OOD)
    'a cartoon avatar with pale skin, short brown hair, glasses and no beard',
    'a cartoon avatar with tan skin, long black hair, no glasses and no beard',
    'a cartoon avatar with light skin, medium ginger hair, sunglasses and a beard',
    'a cartoon avatar with dark skin, short silver hair, no glasses and a beard',
    'a cartoon avatar with dark skin, long black hair, no glasses and no beard',
    'a cartoon avatar with dark skin, long brown hair, glasses and a beard',
    'a cartoon avatar with pale skin, short blonde hair, sunglasses and no beard',
    'a cartoon avatar with tan skin, medium blonde hair, sunglasses and a beard',
]
```

Le prime 4 caption sono nel train (19, 21, 7 e 33 immagini), le ultime 4 solo nell'OOD (30, 43, 35 e 17
immagini): pelle scura con capelli lunghi, capelli biondi con occhiali da sole. Verificato sui file di
`data/`.

## Componenti

### `NoiseScheduler.step(x_t, t, predicted_noise, generator=None)`

- `x_t` `(B, 3, H, W)`, `t` intero Python (lo stesso passo per tutto il batch), `predicted_noise` come `x_t`.
- Con ᾱ = `alpha_bars[t]`, ᾱ_prec = `alpha_bars[t − 1]` (1 se `t = 0`), β = `betas[t]`:
  ```
  x_0 = (x_t − √(1 − ᾱ) · predicted_noise) / √ᾱ        # add_noise turned around
  x_0 = clamp(x_0, −1, 1)                               # real images are in [−1, 1]
  media = √ᾱ_prec · β / (1 − ᾱ) · x_0 + √(1 − β) · (1 − ᾱ_prec) / (1 − ᾱ) · x_t
  t = 0  →  return x_0                                  # a t = 0 la media vale esattamente x_0
  varianza = β · (1 − ᾱ_prec) / (1 − ᾱ)                 # β̃_t
  return media + √varianza · z,   z = torch.randn(..., generator=generator)
  ```
- Restituisce `x_{t−1}`, stessa forma e device di `x_t`.

### `DiffusionModel(config, vocabulary)` (`models/diffusion.py`)

- `text_encoder = TextEncoder(config, vocabulary)` se `TEXT_CONDITIONING`, altrimenti `None`.
- `unet = UNet(config)`.
- `register_buffer('empty_tokens', torch.tensor(vocabulary.encode('')))`: la caption vuota `(L,)`, usata
  dal training (caption dropout) e dal sampling (CFG); segue il modello su GPU e nei checkpoint.
- `forward(x_t, t, tokens)` → rumore predetto `(B, 3, H, W)`: senza text encoder chiama `unet(x_t, t)` e
  ignora `tokens`; altrimenti `context, pad_mask = text_encoder(tokens)` e `unet(x_t, t, context, pad_mask)`.

### `sample(model, scheduler, tokens, config, seed)` (`models/sampling.py`)

Algoritmo 2 del DDPM, sotto `torch.no_grad()`:

```
generator = torch.Generator(device del modello), seed
x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE, generator)     # B = numero di righe di tokens
per t da T − 1 a 0:
    passo = t per tutte le B immagini
    senza text encoder:  predetto = model(x, passo, tokens)
    con text encoder:    con = model(x, passo, tokens)
                         senza = model(x, passo, empty_tokens ripetuta B volte)
                         predetto = senza + GUIDANCE_SCALE · (con − senza)
    x = scheduler.step(x, t, predetto, generator)
return x                                                     # (B, 3, H, W), in [−1, 1]
```

`T` è `scheduler.num_timesteps`; la grandezza delle immagini è `config.IMAGE_SIZE` e la forza della CFG è
`config.GUIDANCE_SCALE` (per provare altri valori nella valutazione si cambia l'attributo di un `Config`).
Il modello va messo in `eval()` da chi chiama.

### `train.py`

- `diffusion_loss(model, scheduler, images, tokens, generator=None)`:
  `t` a caso in 0…T−1 (uno per immagine), rumore gaussiano come `images`, `x_t = scheduler.add_noise(...)`,
  `F.mse_loss(model(x_t, t, tokens), noise)`. `generator` serve alla validation (seed fisso); nel training
  è `None` (generatore globale, inizializzato con `SEED`).
- `update_ema(ema, model, step, config)`: `decay = min(EMA_DECAY, (1 + step) / (10 + step))`; per ogni
  peso `ema = decay · ema + (1 − decay) · peso`.
- `main(config)`, in ordine, stampando cosa fa (come `prepare_data.py`):
  1. `torch.manual_seed(SEED)`; device `'cuda'` se disponibile, altrimenti `'cpu'`.
  2. Cartella dell'esperimento: `RUNS_DIR / nome`, con nome `conditional` o `unconditional`, più
     `_subset{TRAIN_SUBSET}` se `TRAIN_SUBSET > 0`.
  3. `CartoonDataset(config, 'train')` (solo le prime `TRAIN_SUBSET` immagini se > 0) e
     `CartoonDataset(config, 'val')`; `DataLoader` con `BATCH_SIZE`, `shuffle=True` solo per il train.
  4. `DiffusionModel(config, train.vocabulary)` su device; `ema` = copia (`copy.deepcopy`), in `eval()`;
     `NoiseScheduler(config)`; `AdamW(model.parameters(), lr=LEARNING_RATE)`.
  5. Se c'è `last.pt` nella cartella: carica pesi, EMA, optimizer, stato del generatore casuale, e riparte
     dall'epoca successiva.
  6. Per ogni epoca: per ogni batch, caption vuote al 10% (solo con `TEXT_CONDITIONING`), loss,
     `zero_grad`, `backward`, `clip_grad_norm_(GRAD_CLIP)`, `step`, `update_ema`. Poi loss di validation
     con l'EMA e un generatore con seed `SEED`. Poi una riga in `log.csv` (epoca, loss di training media,
     loss di validation, secondi) e la stampa a schermo. Ogni `SAMPLE_EVERY` epoche la griglia
     `samples_epoch{epoca:03d}.png`. Poi `last.pt`.
- `last.pt`: `{'epoch', 'model', 'ema', 'optimizer', 'rng', 'cuda_rng', 'settings'}`; `settings` = tutte le
  impostazioni maiuscole di `Config` (percorsi come stringhe), per sapere con cosa è stato addestrato.
- `if __name__ == '__main__': main(Config())`.

### `ImagePreprocessor.save_preview(images, path)`

Oggi scrive sempre 8 × 8 in `PREVIEW_FILE`. Diventa: 8 colonne, tante righe quante servono per le immagini
ricevute (64 → 8 righe, 16 → 2 righe), nel file `path`. `process_dataset` la chiama con `PREVIEW_FILE`:
la preview dei dati resta identica.

## Come si usa

```
python prepare_data.py          # una volta sola
python train.py                 # TEXT_CONDITIONING = True  → runs/conditional/
python train.py                 # TEXT_CONDITIONING = False → runs/unconditional/
```

Smoke test: `TRAIN_SUBSET = 64` (un batch per epoca), `EPOCHS` alto (per esempio 2000, pochi minuti) e
`SAMPLE_EVERY` più rado: la loss deve scendere molto e la griglia deve somigliare alle 64 immagini.

## Test (`tests/test_training.py`, pytest, senza `data/`)

1. `step` a `t = 0` con il rumore vero restituisce l'immagine pulita
   (`step(add_noise(x_0, 0, ε), 0, ε) ≈ x_0`); a `t = 999` il risultato è finito (niente NaN/inf).
2. `step` con lo stesso seed del generatore dà lo stesso risultato; con seed diversi, risultati diversi.
3. `DiffusionModel`: uscita della forma di `x_t` con e senza testo; senza testo nessun text encoder;
   `empty_tokens` = `vocabulary.encode('')`.
4. `sample` (con `NUM_TIMESTEPS` ridotto a 10 per la velocità): forma `(B, 3, 32, 32)`, valori finiti,
   stesso seed → stesse immagini, seed diversi → immagini diverse; funziona anche senza testo.
5. `diffusion_loss`: un numero; con lo stesso generatore dà lo stesso valore; la backward arriva ai pesi
   del text encoder.
6. `update_ema`: dopo l'aggiornamento ogni peso EMA vale `decay · vecchio + (1 − decay) · modello`, con il
   `decay` dell'avvio graduale.
7. `save_preview` con 16 immagini scrive un PNG di 8 × 2 immagini (in una cartella temporanea).

Verifica finale (non un test automatico): un training brevissimo vero sui dati di `data/`
(`TRAIN_SUBSET = 64`, poche epoche, una griglia) che scrive `last.pt`, `log.csv` e un PNG, e una ripresa
da `last.pt`.

## Fuori da questo lavoro

`generate.py` e demo (prompt + seed → immagine), valutazione (FID/KID, classificatore degli attributi per
le metriche di condizionamento, diversità tra seed, memoria), mixed precision, sampler più veloci (DDIM),
i training lunghi veri (li lanciate voi).

## Modifiche dopo l'implementazione (2026-09-25)

- **Checkpoint atomico** (revisione finale): `last.pt` si scrive su `last.tmp` e poi lo sostituisce in un colpo
  solo (`save_checkpoint`), così un'interruzione durante il salvataggio non rovina il checkpoint precedente.
- **Smoke test con un flag**: `python train.py --smoke-test` al posto di `TRAIN_SUBSET`. Usa la sezione
  `# ---- Smoke test ----` di `config.py` (`SMOKE_TEST_IMAGES = 64`, `SMOKE_TEST_EPOCHS = 8000`,
  `SMOKE_TEST_SAMPLE_EVERY = 1000`, `SMOKE_TEST_CHECKPOINT_EVERY = 100`), cartella `runs/conditional_smoke_test/`.
  La griglia dello smoke test ha in prima riga le prime 8 immagini vere di training e sotto 2 immagini generate
  dalle loro caption.
- **`CHECKPOINT_EVERY`** (1 nel training vero, 100 nello smoke test): validation, checkpoint e riga di log ogni
  N epoche (e all'ultima). Nello smoke test un'epoca è un solo passo: con validation e checkpoint a ogni epoca
  durerebbe ~2,5 ore invece di ~20 minuti. Una riga di `log.csv` riporta la loss di training media dalla riga
  precedente.
- **Ordine a fine epoca**: training → validation → griglia → checkpoint → riga di log. Un'interruzione prima della
  riga fa ripetere epoche, mai righe del log.
- **`DETERMINISTIC`** (`False` di default, vale anche per `--smoke-test`): con `True` la GPU usa solo
  operazioni deterministiche (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`): due training
  dallo stesso seed danno esattamente lo stesso modello (misurato: differenza 0), circa il 30% più lento
  (302 invece di 233 ms per passo). Con `False` differenze minime (0,00007 dopo 30 passi).
