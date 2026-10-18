# Avatar_Diffusion_Model_DL

Piccolo modello di diffusione che genera avatar in stile cartone animato a partire da una caption, con il text encoder
e la UNet addestrati da zero sul Google Cartoon Set. Il modello lavora direttamente
sui pixel, a 32×32 e a 64×64, con 10.000 o 100.000 immagini. La valutazione misura la qualità delle
immagini e il condizionamento sul test ordinario e sulle combinazioni di attributi mai viste insieme durante il
training (OOD).

## Installazione

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Il primo comando installa PyTorch con il supporto per le GPU NVIDIA. Su Colab PyTorch è già installato e basta il
secondo.

## Cartelle

| Cartella                           | Contenuto                                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `dataset/cartoonset100k/`          | i dati originali, con 10.000 immagini nella cartella `0` e tutte le 100.000 nella cartella `100`                               |
| `data/`, `data_100/`               | i dati preparati da 10.000 e da 100.000 immagini (caption, split, vocabolario, immagini a 32×32 e 64×64)                       |
| `runs/`, `runs_100/`               | gli esperimenti con 10.000 e con 100.000 immagini (checkpoint `last.pt`, loss in `log.csv`, griglie di controllo, valutazioni) |
| `runs_longer/`, `runs_100_longer/` | gli addestramenti prolungati                                                                                                   |
| `sample_images/`                   | le figure della relazione                                                                                                      |
| `assets/icons/`                    | le icone della demo                                                                                                            |

## Pipeline

Tutte le impostazioni sono in `config.py`. `IMAGE_SIZE` sceglie la risoluzione (32 o 64) e va cambiata nel file.

```bash
python prepare_data.py                       # caption, split, vocabolario e immagini in data/
python train.py                              # modello di diffusione in runs/conditional_32/
python train_classifier.py                   # classificatore degli attributi in runs/classifier_32/
python guidance_val.py runs/conditional_32   # scelta della guidance sulle caption di validazione
python evaluate.py runs/conditional_32       # valutazione su test e OOD in runs/conditional_32/evaluation/
python app.py                                # demo nel browser, con --share anche un link pubblico
```

- Il vocabolario è costruito solo dalle caption di train. I numeri dello split e le coppie di attributi tenute fuori
  dal training sono in `data/summary.json`.
- Con `IMAGE_SIZE = 64`, `prepare_data.py` aggiunge `data/images_64/` con lo stesso split. `prepare_data_100.py`
  prepara le 100.000 immagini in `data_100/` con le stesse impostazioni.
- Se la cartella dell'esperimento contiene già `last.pt`, `train.py` riprende l'addestramento da lì. Per questo ogni
  esperimento con impostazioni diverse va in una sua `RUNS_DIR`.
- `guidance_val.py` ha scelto la guidance 2, che è il valore di `GUIDANCE_SCALE` usato da `evaluate.py` e proposto
  dalla demo.
  In `runs/conditional_32/` le cartelle `evaluation_guidance1/` ed `evaluation_guidance3/` contengono la stessa
  valutazione con guidance 1 e 3.
- `evaluate.py` confronta le immagini generate con quelle reali della stessa dimensione, quindi `IMAGE_SIZE` deve
  essere quella dell'esperimento.

## Esperimenti

Ogni riga è un esperimento con le impostazioni da scrivere in `config.py`. Quelle che una riga non nomina restano come
nel file consegnato.

| Esperimento                | Impostazioni in `config.py`                                                                                                                                                                                       | Comando                                                                   | Cartella                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Condizionato 32×32, 10.000 | `IMAGE_SIZE = 32`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`                                                                                                                                         | `python train.py`<br>(`python train.py --smoke-test`)                     | `runs/conditional_32`<br>(`runs/conditional_smoke_test_32`)                          |
| Condizionato 64×64, 10.000 | `IMAGE_SIZE = 64`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`                                                                                                                                         | `python train.py`<br>(`python train.py --smoke-test`)                     | `runs/conditional_64`<br>(`runs/conditional_smoke_test_64`)                          |
| Baseline 32×32, 10.000     | `IMAGE_SIZE = 32`, `TEXT_CONDITIONING = False`, `DATA_DIR = ROOT / 'data'`                                                                                                                                        | `python train.py`                                                         | `runs/unconditional_32`                                                              |
| Baseline 64×64, 10.000     | `IMAGE_SIZE = 64`, `TEXT_CONDITIONING = False`, `DATA_DIR = ROOT / 'data'`                                                                                                                                        | `python train.py`                                                         | `runs/unconditional_64`                                                              |
| FiLM 32×32, 10.000         | `IMAGE_SIZE = 32`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`, `RUNS_DIR = ROOT / 'runs' / 'ablations'`                                                                                               | `python train_ablation.py`                                                | `runs/ablations/ablation_pooled/`<br>`conditional_32`                                |
| Prima ablation 64×64       | `IMAGE_SIZE = 64`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`, `UNET_CHANNELS = (64, 128, 256)`, `OFFSET_NOISE = 0.0`, `BATCH_SIZE = 128`, `RUNS_DIR = ROOT / 'runs' / 'ablations' / 'ablation_1_64'` | `python train.py`<br>(`python train.py --smoke-test`)                     | `runs/ablations/ablation_1_64/`<br>`conditional_64`<br>(`conditional_smoke_test_64`) |
| Seconda ablation 64×64     | `IMAGE_SIZE = 64`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`, `OFFSET_NOISE = 0.0`, `RUNS_DIR = ROOT / 'runs' / 'ablations' / 'ablation_2_64'`                                                       | `python train.py`<br>(`python train.py --smoke-test`)                     | `runs/ablations/ablation_2_64/`<br>`conditional_64`<br>(`conditional_smoke_test_64`) |
| Prolungato 32×32, 10.000   | `IMAGE_SIZE = 32`, `TEXT_CONDITIONING = True`, `DATA_DIR = ROOT / 'data'`, `RUNS_DIR = ROOT / 'runs_longer'`, `EPOCHS = 1000`                                                                                     | copiare `runs/conditional_32` in `runs_longer/`,<br>poi `python train.py` | `runs_longer/conditional_32`                                                         |

Con 100.000 immagini sono stati addestrati il condizionato e la baseline a 32×32 e a 64×64, la FiLM 32×32 e il
prolungato 32×32. Si avviano con le impostazioni della riga dello stesso modello, cambiando
`DATA_DIR = ROOT / 'data_100'`, `RUNS_DIR = ROOT / 'runs_100'`, `EPOCHS = 50` e `SAMPLE_EVERY = 5`, e finiscono nelle
stesse cartelle dentro `runs_100/`, tranne la FiLM, che finisce in `runs_100/ablation_pooled/conditional_32`. Il
prolungato invece parte da `runs_100/conditional_32` copiato in `runs_100_longer/`, con
`RUNS_DIR = ROOT / 'runs_100_longer'` ed `EPOCHS = 100`.

Gli addestramenti prolungati riprendono quelli da 500 e da 50 epoche dall'epoca successiva.

Per valutare un esperimento con 100.000 immagini si mette `DATA_DIR = ROOT / 'data_100'` e si rimette
`RUNS_DIR = ROOT / 'runs'`, perché i classificatori degli attributi, sia a 32×32 sia a 64×64, sono in `runs/`.
`--subset` usa tante immagini quante ne ha lo split da 10.000, estratte con `SEED`, così i risultati si confrontano con
quelli dei modelli da 10.000 immagini.

```bash
python evaluate.py runs_100/conditional_32 --subset
```
