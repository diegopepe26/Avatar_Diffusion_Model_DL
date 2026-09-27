# Attribute Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** un classificatore degli attributi (CNN piccola, da zero, una testa per attributo) addestrato sulle immagini vere di train, con la misura della sua affidabilità sulle immagini vere di val, test e OOD: il giudice della metrica di condizionamento.

**Architecture:** `attribute_labels` e `CartoonDataset.labels` trasformano le 5 parole di ogni immagine in 5 numeri di classe, nell'ordine di `MAPPING`. `models/attribute_classifier.py` contiene la rete (3 blocchi convoluzionali, media su tutti i pixel, 5 teste lineari) e `measure_accuracy`, che conta le risposte giuste e servirà anche a `evaluate.py`. `train_classifier.py` addestra per 30 epoche, tiene la migliore su val e scrive `classifier.pt`, `log.csv` e `accuracy.json` in `runs/classifier_<IMAGE_SIZE>/`.

**Tech Stack:** Python 3.12, PyTorch 2.8, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-attribute-classifier-design.md`

## Global Constraints

- Stile del progetto: classi e funzioni brevi, docstring con Args e Returns, commenti brevi in inglese che spiegano il perché, tutte le impostazioni in `config.py`, solo l'essenziale.
- Classi numerate nell'ordine delle parole in `MAPPING`; le teste e le etichette seguono l'ordine degli attributi di `MAPPING` (face_color, hair_color, hair, glasses, facial_hair).
- `CLASSIFIER_CHANNELS = (32, 64, 128)`, `CLASSIFIER_EPOCHS = 30` (50 dopo la modifica del Task 3), `CLASSIFIER_LEARNING_RATE = 1e-3`; batch `BATCH_SIZE` (128); seed `SEED` (42).
- Nessuna data augmentation. Nessuna ripresa dal checkpoint.
- Il file dei pesi si chiama `classifier.pt` (mai `last.pt`), in `runs/classifier_<IMAGE_SIZE>/`.
- `CartoonDataset.__getitem__` non cambia: `train.py` riceve ancora `(immagine, token)`.
- Test: `python -m pytest tests/`, senza `data/` né GPU. Solo test utili (niente mini-dataset finti).
- Branch `attribute-classifier`. Commit solo se `python -m pytest tests/` esce con 0, **in un comando separato** dai test. Ogni commit termina con `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Il push lo fa l'utente.
- Ambiente: niente heredoc né `python -c` su più righe nel Bash (scrivere gli script nella scratchpad); `PYTHONIOENCODING=utf-8` per stampare caratteri non ASCII.

## Review Focus

- **Una parola che diventa il numero sbagliato** (ordine diverso da `MAPPING`) falserebbe tutte le metriche senza errori → Task 1, `test_attribute_labels_follow_the_order_of_mapping`.
- **Immagini a 64×64** (`IMAGE_SIZE = 64`) → stessa rete, stesse forme delle teste → Task 2, `test_one_head_per_attribute_at_32_and_64[64]`.
- **Un'immagine con 4 attributi giusti su 5** non deve contare in `all` → Task 2, `test_measure_accuracy_counts_an_image_only_with_all_five_right`.
- **Le 66 immagini OOD con entrambe le coppie** contano in tutti e due i gruppi di `ood_pairs` (1.120 e 682 immagini) → Task 3, verifica di `accuracy.json`.
- **La demo non deve mostrare `runs/classifier_32/`** tra gli esperimenti → Task 3, verifica con `run_folders`.

---

### Task 1: le etichette (`attribute_labels` e `CartoonDataset.labels`)

**Files:**
- Modify: `preprocessing/cartoon_dataset.py`
- Create: `tests/test_attribute_classifier.py`

**Interfaces:**
- Consumes: `Config.MAPPING` (attributo → parola → valori originali, in quest'ordine: face_color, hair_color, hair, glasses, facial_hair).
- Produces: `attribute_labels(words, config) -> list[int]` (5 interi); `CartoonDataset.labels`: tensore `torch.int64` di forma (N, 5).

- [ ] **Step 1: scrivere il test che fallisce**

Creare `tests/test_attribute_classifier.py`:

```python
"""Tests of the attribute classifier and of its labels. Run them with:  python -m pytest tests/"""

from config import Config
from preprocessing.cartoon_dataset import attribute_labels


def test_attribute_labels_follow_the_order_of_mapping():
    words = {'face_color': 'tan', 'hair_color': 'black', 'hair': 'long', 'glasses': 'no glasses',
             'facial_hair': 'a beard'}
    # the position of each word in its list of MAPPING: tan 1 (dark, tan, ...), black 3, long 3, no glasses 2, a beard 0
    assert attribute_labels(words, Config()) == [1, 3, 3, 2, 0]
```

- [ ] **Step 2: verificare che fallisca**

Run: `python -m pytest tests/test_attribute_classifier.py -v`
Expected: errore di collection, `ImportError: cannot import name 'attribute_labels' from 'preprocessing.cartoon_dataset'`

- [ ] **Step 3: scrivere `attribute_labels` e `CartoonDataset.labels`**

In `preprocessing/cartoon_dataset.py`, prima della classe `CartoonDataset` (dopo gli import), aggiungere:

```python
def attribute_labels(words, config):
    """Turn the five words of an avatar into the numbers of their classes, the answers of the classifier.

    Args:
        words: {attribute: word}, e.g. {'face_color': 'tan', ...}; a row of the captions table works too.
        config: the project Config (uses MAPPING).

    Returns:
        List of 5 integers in the order of MAPPING: the position of each word in its list,
        e.g. tan, black, long, no glasses, a beard -> [1, 3, 3, 2, 0].
    """
    return [list(groups).index(words[attribute]) for attribute, groups in config.MAPPING.items()]
```

Nel docstring di `__init__`, la riga di `config` diventa:

```python
            config: the project Config (uses SPLIT_FILES, VOCABULARY_FILE, RESIZED_IMAGES_DIR, MAPPING).
```

In fondo a `__init__`, dopo la riga di `self.tokens`, aggiungere:

```python
        # the five words of every image as class numbers (N, 5): the answers the attribute classifier must give
        self.labels = torch.tensor([attribute_labels(row, config) for _, row in self.table.iterrows()])
```

- [ ] **Step 4: verificare che passi**

Run: `python -m pytest tests/test_attribute_classifier.py -v`
Expected: 1 passed

- [ ] **Step 5: controllare le etichette sui dati veri**

Scrivere nella scratchpad `check_labels.py`:

```python
from config import Config
from preprocessing.cartoon_dataset import CartoonDataset

dataset = CartoonDataset(Config(), 'train', limit=3)
print(dataset.table[['face_color', 'hair_color', 'hair', 'glasses', 'facial_hair']].to_string())
print(dataset.labels, dataset.labels.dtype, dataset.labels.shape)
```

Run (dalla cartella del progetto): `python <scratchpad>/check_labels.py`
Expected: 3 righe di parole e un tensore (3, 5) `torch.int64` coerente riga per riga con l'ordine di `MAPPING` (per esempio la prima immagine di train, dark / silver / medium / sunglasses / no beard, dà `[0, 4, 2, 1, 1]`).

- [ ] **Step 6: tutti i test**

Run: `python -m pytest tests/`
Expected: 41 passed

- [ ] **Step 7: commit** (solo se lo step 6 è verde)

```bash
git add preprocessing/cartoon_dataset.py tests/test_attribute_classifier.py
git commit -m "feat: CartoonDataset.labels, the five words of every image as class numbers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: la rete e `measure_accuracy`

**Files:**
- Create: `models/attribute_classifier.py`
- Modify: `config.py` (nuova sezione dopo "Smoke test", prima di `to_dict`)
- Modify: `tests/test_attribute_classifier.py`

**Interfaces:**
- Consumes: `Config.MAPPING`, `Config.BATCH_SIZE`; le etichette (N, 5) del Task 1.
- Produces: `AttributeClassifier(config)`, con `forward(images) -> dict[str, Tensor]` (`{attributo: logits (B, numero di parole)}`); `measure_accuracy(classifier, images, labels, config) -> dict[str, float]` (`{attributo: frazione giusta, ..., 'all': frazione con tutti e 5 giusti}`); `Config.CLASSIFIER_CHANNELS`, `Config.CLASSIFIER_EPOCHS`, `Config.CLASSIFIER_LEARNING_RATE`.

- [ ] **Step 1: scrivere i test che falliscono**

In `tests/test_attribute_classifier.py`, gli import in cima diventano:

```python
"""Tests of the attribute classifier and of its labels. Run them with:  python -m pytest tests/"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from models.attribute_classifier import AttributeClassifier, measure_accuracy
from preprocessing.cartoon_dataset import attribute_labels
```

e in fondo al file si aggiunge:

```python
class FixedAnswers(nn.Module):
    """A fake classifier: always the same logits, whatever the images."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.unused = nn.Parameter(torch.zeros(1))   # measure_accuracy reads the device from the parameters

    def forward(self, images):
        return self.logits


@pytest.mark.parametrize('size', [32, 64])
def test_one_head_per_attribute_at_32_and_64(size):
    logits = AttributeClassifier(Config())(torch.randn(2, 3, size, size))
    assert {attribute: tuple(scores.shape) for attribute, scores in logits.items()} == {
        'face_color': (2, 4), 'hair_color': (2, 5), 'hair': (2, 4), 'glasses': (2, 3), 'facial_hair': (2, 2)}


def test_measure_accuracy_counts_an_image_only_with_all_five_right():
    config = Config()
    labels = torch.tensor([[1, 3, 3, 2, 0],     # tan, black, long, no glasses, a beard
                           [3, 1, 2, 0, 1]])    # pale, ginger, medium, glasses, no beard
    answers = labels.clone()
    answers[1, 2] = 1                            # the second image: short hair instead of medium
    # the highest logit on the answer: 1 there, 0 on the other words
    logits = {attribute: F.one_hot(answers[:, i], len(words)).float()
              for i, (attribute, words) in enumerate(config.MAPPING.items())}
    accuracy = measure_accuracy(FixedAnswers(logits), torch.zeros(2, 3, 32, 32), labels, config)
    assert accuracy['hair'] == 0.5
    assert all(accuracy[attribute] == 1 for attribute in ['face_color', 'hair_color', 'glasses', 'facial_hair'])
    assert accuracy['all'] == 0.5                # one image out of two has all five right
```

- [ ] **Step 2: verificare che falliscano**

Run: `python -m pytest tests/test_attribute_classifier.py -v`
Expected: errore di collection, `ModuleNotFoundError: No module named 'models.attribute_classifier'`

- [ ] **Step 3: le impostazioni in `config.py`**

Dopo la sezione `# ---- Smoke test ... ----` (dopo la riga di `SMOKE_TEST_CHECKPOINT_EVERY`) e prima di `def to_dict`, aggiungere:

```python
    # ---- Attribute classifier (python train_classifier.py) ----
    CLASSIFIER_CHANNELS = (32, 64, 128)  # channels of the three blocks; the image halves after each block
    CLASSIFIER_EPOCHS = 30               # the best epoch on val is kept
    CLASSIFIER_LEARNING_RATE = 1e-3      # AdamW, constant
```

- [ ] **Step 4: scrivere `models/attribute_classifier.py`**

```python
"""Attribute classifier: a small CNN that tells the five words of an avatar (skin, hair color, hair, glasses, beard).

It is the judge of the conditioning metric: trained from scratch on the real training images, then frozen,
it reads the generated images and says whether they show the words of their prompt.
"""

import torch
from torch import nn


class AttributeClassifier(nn.Module):
    """Shared convolutions, then one linear head per attribute of MAPPING."""

    def __init__(self, config):
        """Args:
            config: the project Config (uses CLASSIFIER_CHANNELS, MAPPING).
        """
        super().__init__()
        layers = []
        in_channels = 3
        for channels in config.CLASSIFIER_CHANNELS:
            # two 3x3 convolutions, then the image becomes half as big: 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(in_channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
                       nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
                       nn.MaxPool2d(2)]
            in_channels = channels
        self.features = nn.Sequential(*layers)
        # one output per word: face_color 4, hair_color 5, hair 4, glasses 3, facial_hair 2
        self.heads = nn.ModuleDict({attribute: nn.Linear(in_channels, len(words))
                                    for attribute, words in config.MAPPING.items()})

    def forward(self, images):
        """Args:
            images: (B, 3, H, W) in [-1, 1], with H = W = 32 or 64.

        Returns:
            {attribute: (B, number of words) logits}, in the order of MAPPING.
        """
        # the mean over all the pixels: the same number of values at 32x32 and at 64x64
        features = self.features(images).mean(dim=(2, 3))
        return {attribute: head(features) for attribute, head in self.heads.items()}


def measure_accuracy(classifier, images, labels, config):
    """Share of right answers of the classifier, for each attribute and for all five together.

    Used on the real images (train_classifier.py) and on the generated ones (evaluate.py): the two are
    measured in exactly the same way.

    Args:
        classifier: AttributeClassifier (it is left in eval()).
        images: (N, 3, H, W) in [-1, 1], on any device.
        labels: (N, 5) class numbers in the order of MAPPING (see attribute_labels).
        config: the project Config (uses BATCH_SIZE, MAPPING).

    Returns:
        {attribute: fraction of right answers, ..., 'all': fraction of images with all five right}.
    """
    classifier.eval()     # BatchNorm with the statistics of the real training images
    device = next(classifier.parameters()).device
    right = []            # one (batch, 5) tensor per batch: True where the word is right
    with torch.no_grad():
        for start in range(0, len(images), config.BATCH_SIZE):
            logits = classifier(images[start:start + config.BATCH_SIZE].to(device))
            # the word with the highest score, for each attribute: (batch, 5)
            answers = torch.stack([logits[attribute].argmax(dim=1) for attribute in config.MAPPING], dim=1)
            right.append(answers.cpu() == labels[start:start + config.BATCH_SIZE].cpu())
    right = torch.cat(right)
    accuracy = {attribute: right[:, i].float().mean().item() for i, attribute in enumerate(config.MAPPING)}
    accuracy['all'] = right.all(dim=1).float().mean().item()
    return accuracy
```

- [ ] **Step 5: verificare che passino**

Run: `python -m pytest tests/test_attribute_classifier.py -v`
Expected: 4 passed

- [ ] **Step 6: contare i parametri**

Scrivere nella scratchpad `count_parameters.py`:

```python
from config import Config
from models.attribute_classifier import AttributeClassifier

classifier = AttributeClassifier(Config())
print(f'{sum(p.numel() for p in classifier.parameters()):,} parameters')
print(f'{sum(p.numel() for p in classifier.heads.parameters()):,} in the heads')
```

Run: `python <scratchpad>/count_parameters.py`
Expected: circa 290.000 parametri (288.000 nel corpo, 2.322 nelle teste), come nella spec.

- [ ] **Step 7: tutti i test**

Run: `python -m pytest tests/`
Expected: 44 passed

- [ ] **Step 8: commit** (solo se lo step 7 è verde)

```bash
git add config.py models/attribute_classifier.py tests/test_attribute_classifier.py
git commit -m "feat: attribute classifier (small CNN, one head per attribute) and measure_accuracy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `train_classifier.py` e README

Il training non ha un test automatico: l'utente non vuole test con mini-dataset finti. Si verifica lanciandolo sui dati veri a 32×32.

> **Modifica decisa con l'utente durante l'esecuzione** (la spec è aggiornata). Il primo lancio, con il codice
> qui sotto, ha fatto scattare la condizione di stop dello step 3: tutti e 5 giusti nel 97,5% del test ma solo
> nell'88,7% delle OOD vere, con errori che seguono le coppie (dark → tan, blonde → altro colore, sunglasses →
> glasses). L'utente ha scelto di addestrare il classificatore anche su metà delle immagini vere OOD e di portare
> le epoche a 50. Nel codice finale: le OOD si mescolano con `torch.randperm` e un generatore con `SEED`; la prima
> metà (868) si aggiunge al train del classificatore, la seconda (868) è la metà di controllo, su cui si misurano
> `ood` e `ood_pairs`; `accuracy.json` ha in più `ood_train_images`; `CLASSIFIER_EPOCHS = 50` in `config.py`; il
> README descrive la metà OOD. I conteggi attesi allo step 3 diventano: `ood.images` 868 e le coppie sulla metà di
> controllo (569 e 330 con `SEED` = 42).

**Files:**
- Create: `train_classifier.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `CartoonDataset(config, split)` con `.images` (lista di tensori (3, H, W)), `.labels` (N, 5), `.table` (con la colonna `ood_pair` nello split `ood`); `AttributeClassifier(config)`; `measure_accuracy(classifier, images, labels, config)`.
- Produces: `runs/classifier_<IMAGE_SIZE>/classifier.pt` = `{'weights': state_dict, 'settings': {'IMAGE_SIZE': int, 'CLASSIFIER_CHANNELS': tuple}, 'epoch': int, 'val_accuracy': float}`; `log.csv` con le colonne `epoch, train_loss, train_accuracy, val_accuracy`; `accuracy.json` con le chiavi `epoch`, `val`, `test`, `ood`, `ood_pairs` (nomi come in `summary.json`, per esempio `'dark + long'`). `evaluate.py` userà `classifier.pt` e `accuracy.json`.

- [ ] **Step 1: scrivere `train_classifier.py`**

```python
"""Trains the attribute classifier, the judge of the conditioning metric, on the real training images.

Run it with:  python train_classifier.py    (after prepare_data.py; the size is IMAGE_SIZE in config.py)
The results go to runs/classifier_<IMAGE_SIZE>/: classifier.pt (the best epoch on val), log.csv, and
accuracy.json (how often it is right on the real val, test and OOD images).
"""

import csv
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from models.attribute_classifier import AttributeClassifier, measure_accuracy
from preprocessing.cartoon_dataset import CartoonDataset


def classifier_loss(logits, labels, config):
    """Sum of the cross-entropies of the five heads, all with the same weight.

    Args:
        logits: {attribute: (B, number of words)}, from AttributeClassifier.
        labels: (B, 5) class numbers in the order of MAPPING.
        config: the project Config (uses MAPPING).

    Returns:
        Tensor with one number.
    """
    return sum(F.cross_entropy(logits[attribute], labels[:, i]) for i, attribute in enumerate(config.MAPPING))


def main(config):
    """Train for CLASSIFIER_EPOCHS epochs, keep the best epoch on val, then measure it on val, test and OOD.

    Args:
        config: the project Config (uses the Attribute classifier section, BATCH_SIZE, SEED, IMAGE_SIZE,
            OOD_PAIRS, RUNS_DIR).
    """
    print('Training of the attribute classifier')
    torch.manual_seed(config.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. One classifier per image size: runs/classifier_32, runs/classifier_64
    run_dir = config.RUNS_DIR / f'classifier_{config.IMAGE_SIZE}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'1. Folder: {run_dir} (device: {device})')

    # 2. The real images and their five words as class numbers
    data = {split: CartoonDataset(config, split) for split in ['train', 'val', 'test', 'ood']}
    images = {split: torch.stack(dataset.images) for split, dataset in data.items()}
    train_loader = DataLoader(TensorDataset(images['train'], data['train'].labels),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    print('2. Data: ' + ', '.join(f'{split} {len(dataset)} images' for split, dataset in data.items()))

    # 3. The classifier and the optimizer
    classifier = AttributeClassifier(config).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.CLASSIFIER_LEARNING_RATE)
    print(f'3. Classifier: {sum(p.numel() for p in classifier.parameters()):,} parameters')

    # 4. Training: after every epoch the accuracy on train and on val, the best epoch on val is saved
    checkpoint_file = run_dir / 'classifier.pt'
    log_file = run_dir / 'log.csv'
    with open(log_file, 'w', newline='') as f:     # a new log at every run
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_accuracy', 'val_accuracy'])
    best = -1.0
    print(f'4. Epochs 1-{config.CLASSIFIER_EPOCHS}')
    for epoch in range(1, config.CLASSIFIER_EPOCHS + 1):
        start = time.time()
        classifier.train()     # measure_accuracy leaves it in eval(): BatchNorm must learn again
        total = 0.0
        for batch, labels in train_loader:
            loss = classifier_loss(classifier(batch.to(device)), labels.to(device), config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        # both in eval(), so they can be compared: a growing gap means overfitting
        train_accuracy = measure_accuracy(classifier, images['train'], data['train'].labels, config)['all']
        val_accuracy = measure_accuracy(classifier, images['val'], data['val'].labels, config)['all']
        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{train_loss:.6f}', f'{train_accuracy:.4f}', f'{val_accuracy:.4f}'])
        saved = ''
        if val_accuracy > best:     # strictly better: on a tie the older epoch stays
            best = val_accuracy
            torch.save({'weights': classifier.state_dict(),
                        'settings': {'IMAGE_SIZE': config.IMAGE_SIZE,
                                     'CLASSIFIER_CHANNELS': config.CLASSIFIER_CHANNELS},
                        'epoch': epoch,
                        'val_accuracy': val_accuracy}, checkpoint_file)
            saved = ', saved'
        print(f'   epoch {epoch}/{config.CLASSIFIER_EPOCHS}: train loss {train_loss:.4f}, '
              f'train accuracy {train_accuracy:.4f}, val accuracy {val_accuracy:.4f}, '
              f'{time.time() - start:.0f} s{saved}')

    # 5. The best epoch on the real images of every split and of every held-out pair
    checkpoint = torch.load(checkpoint_file, map_location=device)
    classifier.load_state_dict(checkpoint['weights'])
    report = {'epoch': checkpoint['epoch']}
    for split in ['val', 'test', 'ood']:
        report[split] = measure_accuracy(classifier, images[split], data[split].labels, config)
        report[split]['images'] = len(data[split])
    report['ood_pairs'] = {}
    for pair in config.OOD_PAIRS:
        name = ' + '.join(pair.values())     # as in the column ood_pair, e.g. 'dark + long'
        # contains, not ==: an image with both pairs has 'dark + long; blonde + sunglasses' and counts in both
        rows = torch.tensor(data['ood'].table['ood_pair'].str.contains(name, regex=False).to_numpy())
        report['ood_pairs'][name] = measure_accuracy(classifier, images['ood'][rows], data['ood'].labels[rows],
                                                     config)
        report['ood_pairs'][name]['images'] = int(rows.sum())
    with open(run_dir / 'accuracy.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'5. Best epoch {checkpoint["epoch"]}: all five words right in {report["test"]["all"]:.1%} of test, '
          f'{report["ood"]["all"]:.1%} of OOD (accuracy.json)')
    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main(Config())
```

- [ ] **Step 2: lanciare il training vero a 32×32**

Controllare che `config.py` abbia `IMAGE_SIZE = 32`, poi:

Run: `python train_classifier.py` (in background se serve; stimato circa un minuto sulla RTX 4060)
Expected:
- `2. Data: train 5754 images, val 1255 images, test 1255 images, ood 1736 images`;
- `3. Classifier:` circa 290.000 parametri;
- 30 righe di epoca; la val accuracy sale nelle prime epoche; almeno una riga con `, saved`;
- la riga 5 e `Done: ...runs\classifier_32`.

- [ ] **Step 3: controllare i file prodotti**

- `runs/classifier_32/log.csv`: intestazione e 30 righe.
- `runs/classifier_32/accuracy.json`: `val.images` 1255, `test.images` 1255, `ood.images` 1736, `ood_pairs["dark + long"].images` 1120, `ood_pairs["blonde + sunglasses"].images` 682; ogni gruppo ha i 5 attributi e `all`.
- Annotare, per riportarli all'utente: l'epoca migliore, `all` su val, test e OOD, gli attributi delle due coppie (`face_color` e `hair` in "dark + long", `hair_color` e `glasses` in "blonde + sunglasses"), la distanza tra `train_accuracy` e `val_accuracy` nelle ultime epoche, il tempo totale.
- Se l'accuratezza di una coppia OOD è molto più bassa di quella di test, **fermarsi e segnalarlo all'utente**: è il caso in cui la spec prevede il piano di riserva (5 reti separate).
- Se `train_accuracy` e `val_accuracy` si allontanano, segnalarlo: è il caso in cui la spec rimette in discussione l'augmentation.

- [ ] **Step 4: la demo non vede il classificatore**

Run: `python -c "from config import Config; from generate import run_folders; print([f.name for f in run_folders(Config())])"`
Expected: la lista non contiene `classifier_32` (contiene `conditional_32`).

- [ ] **Step 5: README**

In `README.md`, nella tabella "Code", la riga di `cartoon_dataset.py` diventa:

```markdown
| `preprocessing/cartoon_dataset.py` | `CartoonDataset`: PyTorch Dataset of one split (images, caption ids, `labels` = the five words as class numbers); `attribute_labels`: words -> class numbers |
```

e, dopo la riga di `app.py`, si aggiungono:

```markdown
| `models/attribute_classifier.py` | `AttributeClassifier`: small CNN, one head per attribute; `measure_accuracy`: share of right words, per attribute and all five together |
| `train_classifier.py` | trains the attribute classifier on the real training images, keeps the best epoch on val, measures it on the real val, test and OOD images |
```

Tra la sezione "Demo" e la sezione "Tests" si aggiunge (con il tempo misurato allo step 2 al posto di "about a minute", se diverso):

````markdown
## Attribute classifier

The judge of the conditioning metric: FID and KID say whether the generated images look like real avatars, not
whether they show the words of their prompt. A small CNN (about 290,000 parameters: three convolutional blocks,
the mean over all the pixels, one linear head per attribute) is trained from scratch on the real training images
and tells the five words of an avatar. After `prepare_data.py`:

```bash
python train_classifier.py
```

About a minute at 32x32 on an RTX 4060 Laptop. It works at `IMAGE_SIZE` (one classifier per size) and writes to
`runs/classifier_32/` (`_64` with `IMAGE_SIZE = 64`):

| File | Content |
|---|---|
| `classifier.pt` | weights of the best epoch on val (share of images with all five words right), its settings, epoch and val accuracy |
| `log.csv` | one row per epoch: train loss, accuracy on train and on val (all five right); a growing gap between the two means overfitting |
| `accuracy.json` | share of right words on the real val, test and OOD images, and on each held-out pair: the ceiling of the conditioning metric |

The weights are in `classifier.pt` and not in `last.pt`, so the demo does not list the classifier among the
experiments. The training does not resume: run it again to start from scratch.
````

- [ ] **Step 6: tutti i test**

Run: `python -m pytest tests/`
Expected: 44 passed

- [ ] **Step 7: commit** (solo se lo step 6 è verde)

```bash
git add train_classifier.py README.md
git commit -m "feat: train_classifier.py trains the attribute classifier and measures it on the real images

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: riportare all'utente** i numeri annotati allo step 3 e il comando per il push: `git push -u origin attribute-classifier`.
