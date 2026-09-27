# Generazione e demo web: design

Data: 2026-09-27 · Branch: `generate` (da `unet-64`)

## Obiettivo

Una pagina web locale per generare avatar con un modello addestrato, senza scrivere comandi nel terminale:
si scelgono i 5 attributi da menu, il seed, il numero di immagini, la guidance e l'esperimento; le immagini
compaiono nella pagina e si salvano nella cartella dell'esperimento. È la demo chiesta dalla traccia ("enter a
prompt, choose a seed, and inspect the generated 32x32 or 64x64 image"). Le funzioni di generazione stanno in
un file a parte, riusabile da `evaluate.py`.

## Decisioni

- **Gradio**: libreria Python per demo di machine learning; `python app.py` apre la pagina nel browser e il
  modello gira sulla GPU locale; su Colab `python app.py --share` crea un link pubblico (utile all'orale).
  Nuova dipendenza `gradio` in `requirements.txt`.
- **Un seed per immagine**: con seed 7 e 4 immagini si generano i seed 7, 8, 9, 10, **ognuno con il suo
  generatore casuale**: "prompt + seed 8" dà sempre la stessa immagine, da sola o in gruppo. Si ottiene
  chiamando il `sample()` esistente una volta per immagine (batch 1): **nessuna modifica** a `sample()`,
  `step()` e `train.py`. Misurato sul checkpoint vero a 32 × 32: 19 s per immagine con la CFG, e due
  generazioni con lo stesso prompt e seed identiche al bit. La griglia di controllo del training resta com'è (un
  generatore per tutta la griglia: serve solo a confrontare le epoche).
- **Prompt dai menu**: i menu si riempiono con le parole di `Config.MAPPING`; il prompt si compone con
  `CaptionGenerator.compose_caption` (lo stesso `TEMPLATE` del training), mostrato nella pagina. Solo parole
  note al modello, sempre nella forma del training.
- **Avviso OOD**: se gli attributi scelti contengono una coppia di `Config.OOD_PAIRS` (pelle scura + capelli
  lunghi, capelli biondi + occhiali da sole), la pagina scrive che la combinazione non è mai stata vista nel
  training.
- **Esperimento da menu**: le cartelle di `RUNS_DIR` che contengono `last.pt`. La rete si ricostruisce con le
  impostazioni **salvate nel checkpoint** (`IMAGE_SIZE`, `TEXT_CONDITIONING`, `D_MODEL`, `NUM_HEADS`,
  `NUM_ENCODER_BLOCKS`, `FFN_DIM`, `DROPOUT`, `ATTENTION`, `UNET_CHANNELS`, `TIME_DIM`, `NUM_TIMESTEPS`), non
  con quelle attuali di `config.py`: un esperimento a 32 funziona anche con `IMAGE_SIZE = 64` in `config.py`.
  Pesi EMA; vocabolario da `data/vocabulary.json`. Ogni esperimento si carica una volta e resta in memoria.
- **Baseline** (esperimento senza testo): gli attributi e la guidance si ignorano (la pagina lo scrive).
- **Parametri**: seed (intero, predefinito 0), numero di immagini (1-8, predefinito 4), guidance (da 1 a 7 a
  passi di 0,5, predefinito `GUIDANCE_SCALE` = 3).
- **Salvataggio** (casella, attiva di default): un PNG per immagine, ingrandito 4 volte come le griglie, in
  `runs/<esperimento>/generated/`, con nome dagli attributi e dal seed, per esempio
  `pale_long_blonde_no-glasses_a-beard_seed7.png` (spazi → trattini). La baseline: `unconditional_seed7.png`.
- **Nella pagina**: le immagini con il loro seed, il tempo impiegato, la cartella di salvataggio.

## Vincoli di stile

Come il resto del progetto: funzioni brevi con docstring Args/Returns, commenti in inglese sul *perché*,
solo l'essenziale. `generate.py` contiene la logica (niente interfaccia); `app.py` solo la pagina.

## File

```
generate.py        load_model(run_dir), generate(...), is_held_out(words, config), run_folders(config)
app.py             la pagina Gradio: menu, parametri, pulsante, galleria, salvataggio; --share per Colab
requirements.txt   + gradio
README.md          sezione "Demo" e righe nella tabella "Code"
tests/test_generate.py
```

### `generate.py`

- `load_model(run_dir)`: legge `last.pt`, copia in un `Config` le impostazioni della rete salvate nel
  checkpoint, carica il vocabolario, costruisce `DiffusionModel` con i pesi EMA in `eval()` sul device
  (GPU se c'è), e il `NoiseScheduler`. Restituisce `(model, scheduler, config, vocabulary)`.
- `generate(model, scheduler, config, vocabulary, caption, seeds, guidance)`: per ogni seed, `sample()` su
  un'immagine con quel seed e con `GUIDANCE_SCALE = guidance`; restituisce le immagini come PIL (valori 0-255,
  dimensione nativa), nello stesso ordine dei seed.
- `is_held_out(words, config)`: vero se `words` contiene una coppia di `OOD_PAIRS`.
- `run_folders(config)`: le sottocartelle di `RUNS_DIR` con un `last.pt`, in ordine alfabetico.

## Test (`tests/test_generate.py`, pytest, senza `data/` né GPU)

Con un checkpoint piccolo creato nel test (modello non addestrato, `NUM_TIMESTEPS = 10` per la velocità,
vocabolario costruito nel test):
1. **Un seed per immagine**: l'immagine con seed 8 generata nel gruppo (7, 8) è identica a quella generata da
   sola con seed 8.
2. **Impostazioni dal checkpoint**: un checkpoint salvato con `IMAGE_SIZE = 64` dà una rete a 64 (immagini
   64 × 64) anche se `config.py` dice 32.
3. **`is_held_out`**: vero per pelle scura + capelli lunghi, falso per una combinazione vista.

La pagina (`app.py`) si verifica a mano: si apre, genera, mostra e salva.

## Fuori da questo lavoro

`evaluate.py` e le metriche, il classificatore degli attributi, una generazione più veloce a gruppi con un
generatore per immagine.
