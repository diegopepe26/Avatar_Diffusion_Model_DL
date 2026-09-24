# Noise scheduler: design

Data: 2026-09-24 · Branch: `noise-scheduler` (da `text-encoder`)

## Obiettivo

Implementare il **forward diffusion**: la ricetta fissa, senza pesi, che porta un'immagine pulita al livello di
rumore del passo `t`. Nel training servirà a creare gli esercizi per la UNet: dato `x_t` e `t`, predire il
rumore aggiunto. La UNet, il training e la generazione non fanno parte di questo lavoro.

## Decisioni

- **Curva cosine** (Nichol & Dhariwal 2021, *Improved DDPM*), non lineare (DDPM). Con la lineare e 1000
  passi, 327 passi su 1000 lasciano meno dell'1% dell'immagine (rumore quasi puro, niente da imparare);
  con la cosine solo 65. A metà strada (`t = 500`) la lineare lascia l'8% dell'immagine, la cosine il 49%.
- **`T = 1000` passi**, `t` da 0 a 999 (già `t = 0` ha un pizzico di rumore). Valore dei due paper di
  riferimento. Il costo del training non dipende da T (un solo `t` a caso per immagine); T pesa solo sulla
  generazione (T passaggi nella UNet). Cambiare T richiede di riaddestrare.
- **Scorciatoia**: nel training non si percorrono i passi uno per uno. Mille piccole aggiunte di rumore
  gaussiano equivalgono a una sola, di intensità nota:
  `x_t = √ᾱ_t · immagine + √(1 − ᾱ_t) · rumore`, dove ᾱ_t = quanta immagine resta dopo il passo `t`.
- **Il rumore si aggiunge nel ciclo di training** (per batch, su GPU), non in `CartoonDataset`, che resta
  invariato e continua a restituire immagini pulite in [-1, 1].
- **`t` e il rumore si estraggono fuori dallo scheduler** (in `train.py`) e si passano ad `add_noise`:
  la loss ha bisogno dello stesso rumore come risposta giusta, perché la UNet predirà il rumore.
- **Il time embedding non riguarda lo scheduler**: `sinusoidal_embedding(t, dim)` si userà dentro la UNet,
  perché la rete sappia quanto rumore c'è. Lo scheduler usa `t` solo come indice della sua tabella.

## Vincoli di stile

Come il resto del progetto (vedi `preprocessing/` e `models/`): classe semplice, `config` nel costruttore,
docstring con Args/Returns, commenti brevi in inglese sul *perché*, solo l'essenziale. Per ora la classe ha
`__init__` e `add_noise`; il metodo per togliere un passo di rumore arriverà con la generazione.

## File

```
config.py                       + sezione "Diffusion"
models/__init__.py              docstring aggiornata (non solo reti: anche lo scheduler)
models/noise_scheduler.py       NoiseScheduler
tests/test_noise_scheduler.py   test con pytest
README.md                       una riga nella tabella "Code"
```

## Configurazione (`config.py`)

```python
# ---- Diffusion ----
NUM_TIMESTEPS = 1000      # T: noise steps, t = 0 (almost clean) ... T - 1 (pure noise)
```

Le costanti della formula cosine (`s = 0.008`, beta massimo `0.999`) restano nel codice con un commento:
sono parte della formula del paper, come il 10000 in `sinusoidal_embedding`, non impostazioni da regolare.

## `NoiseScheduler(config)` (`models/noise_scheduler.py`)

Non è un `nn.Module`: non ha pesi da addestrare e non va spostato su GPU.

### `__init__(config)`

- `T = config.NUM_TIMESTEPS`
- Curva cosine: `ᾱ_cos(x) = cos²((x + 0.008) / 1.008 · π/2)`, con `x` da 0 a 1.
- Per ogni passo `t = 0 … T−1`: `beta_t = min(1 − ᾱ_cos((t+1)/T) / ᾱ_cos(t/T), 0.999)`.
  Il limite 0.999 evita l'ultimo passo con beta = 1 (ᾱ esattamente 0), che in generazione darebbe una
  divisione per zero.
- `self.betas`: tensore `(T,)`, rumore aggiunto dal singolo passo `t` (servirà alla generazione).
- `self.alpha_bars`: tensore `(T,)`, `alpha_bars[t] = (1 − beta_0) · (1 − beta_1) · … · (1 − beta_t)`,
  quanta immagine resta dopo il passo `t`.
- Entrambi float32, sulla CPU.

Valori attesi (calcolati in fase di design): `alpha_bars[0] ≈ 1`, `alpha_bars[499] ≈ 0.49`,
`alpha_bars[999] ≈ 0`; `betas` crescenti da ~0.00004 a 0.999.

### `add_noise(images, t, noise)`

- Input: `images` `(B, 3, H, W)` in [-1, 1]; `t` `(B,)` interi tra 0 e T−1, uno per immagine;
  `noise` `(B, 3, H, W)`, rumore gaussiano della stessa forma.
- `ᾱ = alpha_bars` portato sul device di `images`, indicizzato con `t`, messo in forma `(B, 1, 1, 1)`
  così vale per tutti i canali e i pixel della sua immagine.
- Output: `x_t = √ᾱ · images + √(1 − ᾱ) · noise`, forma `(B, 3, H, W)`, sullo stesso device.

## Uso previsto nel training (riferimento, non implementato qui)

```
t = interi a caso tra 0 e T−1, uno per immagine    (torch.randint)
noise = rumore gaussiano come images               (torch.randn_like)
x_t = scheduler.add_noise(images, t, noise)
predetto = unet(x_t, t, context, pad_mask)
loss = MSE(predetto, noise)
```

## Test (`tests/test_noise_scheduler.py`, pytest)

Nessun file da `data/`.

1. Curva: `alpha_bars` e `betas` hanno 1000 valori; `alpha_bars[0] > 0.999`, `alpha_bars` strettamente
   decrescente, `alpha_bars[499] ≈ 0.49`, `alpha_bars[999] < 0.001`; `betas` positivi e al massimo 0.999.
2. `add_noise` agli estremi: con `t = 0` il risultato è quasi l'immagine pulita, con `t = 999` quasi solo
   il rumore.
3. `add_noise` con un `t` diverso per ogni immagine del batch: ogni immagine riceve il suo ᾱ (confronto con
   la formula calcolata immagine per immagine).
4. Su GPU, se disponibile (altrimenti il test si salta): immagini, `t` e rumore su CUDA, risultato su CUDA,
   senza spostare lo scheduler.

Esecuzione: `python -m pytest tests/`. I test si scrivono prima del codice (TDD).

## Fuori da questo lavoro

UNet (con il time embedding), generazione (metodo per togliere un passo di rumore), `train.py`,
classifier-free guidance, sezione training di `config.py` (`BATCH_SIZE`, learning rate, epoche).
