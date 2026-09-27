# Classificatore degli attributi: design

Data: 2026-09-27 · Branch: `attribute-classifier` (da `smoke-test-images`)

## Obiettivo

Un "giudice" automatico per la **metrica di condizionamento** chiesta dalla traccia ("both quality **and
conditioning** metrics"). FID e KID dicono se le immagini generate sembrano avatar veri, ma non se sono l'avatar
chiesto dal prompt. Il classificatore guarda un'immagine e dice quale parola vede per ognuno dei 5 attributi.
Confrontando la sua risposta con le parole del prompt si ottiene la percentuale di attributi rispettati.

Il classificatore è uno strumento di valutazione, come l'Inception del FID: non fa parte del modello di diffusione.
Non è un discriminatore come nelle GAN. Si addestra prima e da solo, sulle immagini vere, poi resta fermo e serve
solo a dare i voti.

Questo lavoro comprende la rete, lo script che la addestra e la misura della sua affidabilità sulle immagini vere.
`evaluate.py`, che la userà sulle immagini generate, è il pezzo successivo.

## Decisioni

- **Un corpo comune e 5 teste**, una per attributo (4, 5, 4, 3 e 2 classi). Una sola testa con tutte le
  4 × 4 × 5 × 3 × 2 = 480 combinazioni non potrebbe mai rispondere con una combinazione OOD, perché nel train non
  c'è. Cinque reti separate costerebbero 5 volte tanto: ognuna imparerebbe da capo le stesse cose di base (dove
  sono la faccia, i bordi, i colori). Le teste pesano meno dell'1% dei parametri, quindi la specializzazione per
  attributo costa pochissimo.
- **Rischio di scorciatoia.** Nel train "dark" e "long" (e "blonde" e "sunglasses") non compaiono mai insieme, e
  la rete potrebbe imparare "se è dark, i capelli non sono long". La correlazione è nei dati, quindi nessuna
  architettura la elimina da sola.
- **Il classificatore si addestra anche su metà delle immagini vere OOD** (decisione presa dopo il primo training,
  vedi sotto). Il primo training, solo sul train, indovinava tutti e 5 gli attributi nel 97,5% delle immagini vere
  di test e solo nell'88,7% delle OOD (88,4% su "dark + long", 88,0% su "blonde + sunglasses"). Gli errori erano
  proprio la scorciatoia: su "dark + long" 29 volte tan invece di dark e 14 volte medium invece di long; su
  "blonde + sunglasses" 30 volte un altro colore invece di blonde e 40 volte occhiali normali o nessun occhiale
  invece di sunglasses. Con quel giudice, sulle immagini generate OOD non si potrebbero distinguere gli errori del
  modello di diffusione da quelli del giudice. Il classificatore è uno strumento di misura, come l'Inception del
  FID, e non fa parte della generazione: la regola della traccia (combinazioni OOD assenti dal training) vale per
  il modello di diffusione, che continua a non vederle mai. Le 1.736 immagini OOD si mescolano con `SEED` e si
  dividono a metà: 868 si aggiungono al train del classificatore, le altre 868 servono a misurarne l'affidabilità
  sulle combinazioni OOD. Lo split del modello di diffusione (`data/captions_*.csv`) non cambia.
- **Un classificatore per risoluzione**, che segue `IMAGE_SIZE` come il resto del progetto: si addestra su
  `data/images_32/` o `data/images_64/` e si salva in `runs/classifier_32/` o `runs/classifier_64/`. Rimpicciolire
  le immagini a 64 per giudicarle a 32 cancellerebbe dettagli fini (la montatura degli occhiali); ingrandire quelle
  a 32 le renderebbe sfocate e diverse da quelle del training del classificatore.
- **Nessuna data augmentation.** Le immagini sono pulite e regolari (sfondo bianco, faccia sempre nello stesso
  posto) e anche le classi più rare hanno centinaia di esempi nel train (balding 316, black 599). Il ribaltamento
  orizzontale non creerebbe combinazioni nuove, quindi non aiuterebbe sulle OOD. I cambi di colore sono esclusi in
  ogni caso, perché il colore è l'etichetta. L'overfitting si controlla nel `log.csv` (accuratezza su train e su
  val a ogni epoca): se le due si allontanano, se ne riparla.
- **50 epoche, si tiene la versione migliore su val.** Il criterio è la percentuale di immagini di val con tutti e
  5 gli attributi giusti. Test e la metà di controllo delle OOD non si guardano mai durante il training. Erano 30:
  nel primo training la curva su val andava su e giù (68% all'epoca 23) e la versione migliore era l'ultima, quindi
  il training non era ancora stabile.
- **Il file si chiama `classifier.pt`, non `last.pt`**, altrimenti la demo (`run_folders` in `generate.py`) lo
  mostrerebbe tra gli esperimenti di diffusione.
- **Nessuna ripresa dal checkpoint.** Il training dura pochi minuti: se si interrompe, si rilancia da capo e i file
  si sovrascrivono.

## La rete (`models/attribute_classifier.py`)

```
immagine (3, 32, 32), valori in [-1, 1]
  blocco 1: conv 3×3 → BatchNorm → ReLU → conv 3×3 → BatchNorm → ReLU → MaxPool   → (32, 16, 16)
  blocco 2: uguale                                                               → (64, 8, 8)
  blocco 3: uguale                                                               → (128, 4, 4)
  media su tutti i pixel (global average pooling)                                → 128 numeri
  5 teste lineari: face_color 4 · hair_color 5 · hair 4 · glasses 3 · facial_hair 2
```

- Canali `CLASSIFIER_CHANNELS = (32, 64, 128)`: circa 290.000 parametri (287.000 nelle convoluzioni, 2.300 nelle
  teste), circa 20 volte meno della U-Net (6,1 milioni).
- A 64×64 i blocchi sono gli stessi e l'uscita del blocco 3 è (128, 8, 8); la media finale la riporta a 128
  numeri, quindi le teste non cambiano.
- Receptive field di 36 pixel: a 32×32 ogni numero finale vede tutta l'immagine; a 64×64 ne vede un pezzo di
  36×36, e la media raccoglie le informazioni di tutti i pezzi. Se a 64 l'accuratezza su val fosse bassa, si
  aggiungerebbe un blocco (non previsto ora).
- BatchNorm e ReLU, la scelta standard per i classificatori con batch grandi. In `eval()` la BatchNorm usa le
  medie calcolate sulle immagini vere, quindi le immagini generate vengono misurate con lo stesso metro.
- `forward(images)` restituisce un dizionario `{attributo: logits}`, nell'ordine di `MAPPING`; per esempio
  `logits['hair']` ha forma (B, 4).
- Le classi sono numerate nell'ordine delle parole in `MAPPING`: per `face_color` dark = 0, tan = 1, light = 2,
  pale = 3. Il numero è una categoria, non una misura: scambiare dark con tan è un errore quanto scambiarlo con
  pale.
- La predizione di un'immagine è l'argmax di ogni testa: un vettore di 5 numeri, per esempio `[1, 3, 3, 2, 0]` =
  tan, black, long, no glasses, a beard.

`measure_accuracy(classifier, images, labels, config)` sta nello stesso file, perché la userà anche
`evaluate.py`. Riceve le immagini (N, 3, H, W) in [-1, 1] e le etichette (N, 5); le passa al classificatore in
`eval()` e senza gradienti, a gruppi di `BATCH_SIZE`, sul device del classificatore. Restituisce
`{attributo: frazione giusta, ..., 'all': frazione con tutti e 5 giusti}`. Sulle immagini vere ora, e su quelle
generate nel pezzo successivo, si misura così esattamente nello stesso modo.

## Le etichette (`preprocessing/cartoon_dataset.py`)

- `attribute_labels(words, config)`: da `{attributo: parola}` (va bene anche una riga della tabella) alla lista
  dei 5 numeri di classe, nell'ordine di `MAPPING`. Per esempio "tan, black, long, no glasses, a beard" →
  `[1, 3, 3, 2, 0]`.
- `CartoonDataset` riceve il campo `self.labels`: tensore intero (N, 5), costruito con `attribute_labels` dalle
  colonne della tabella. `__getitem__` non cambia, quindi `train.py` riceve ancora `(immagine, token)`.
- Nel pezzo successivo `evaluate.py` caricherà `CartoonDataset(config, 'test')` e avrà insieme le immagini vere
  (per il FID), i token (per generare) e le etichette attese (per l'accuratezza).

## Il training (`train_classifier.py`)

`python train_classifier.py`, dopo `prepare_data.py`. Lavora alla risoluzione di `IMAGE_SIZE`.

1. Seed fisso (`SEED` = 42), device GPU se c'è. Legge train, val, test e OOD con `CartoonDataset`. Mescola le
   immagini OOD con un generatore inizializzato con `SEED` e le divide a metà: la prima metà (868) si aggiunge al
   train del classificatore (5.754 + 868 = 6.622 immagini), la seconda (868) è la metà di controllo.
2. `AttributeClassifier`, AdamW con learning rate `CLASSIFIER_LEARNING_RATE` = 10⁻³ costante, batch `BATCH_SIZE`
   = 128, immagini mescolate a ogni epoca.
3. La loss è la somma delle 5 cross-entropy, con lo stesso peso.
4. Per `CLASSIFIER_EPOCHS` = 50 epoche:
   - una passata sul train del classificatore (train + metà OOD);
   - `measure_accuracy` su tutto il train del classificatore e su val (in `eval()`, quindi le due accuratezze sono
     confrontabili);
   - una riga in `log.csv`: `epoch, train_loss, train_accuracy, val_accuracy` (le accuratezze sono il valore
     `all`); `train_loss` è la media delle loss dell'epoca;
   - una riga nel terminale con gli stessi numeri e il tempo;
   - se `val_accuracy` è strettamente maggiore della migliore finora, salva `classifier.pt` (a parità resta la
     versione più vecchia).
5. Alla fine ricarica la versione migliore, la misura su val, test e sulla metà di controllo delle OOD e scrive
   `accuracy.json`.

Il primo training (30 epoche) è durato circa 30 secondi a 32×32 sulla RTX 4060; con 50 epoche e 868 immagini in
più si stima circa un minuto, qualche minuto a 64×64.

### I file in `runs/classifier_<IMAGE_SIZE>/`

| File | Contenuto |
|---|---|
| `classifier.pt` | `weights` (i pesi), `settings` (`IMAGE_SIZE`, `CLASSIFIER_CHANNELS`: servono a ricostruire la rete), `epoch` e `val_accuracy` della versione migliore |
| `log.csv` | una riga per epoca; si riscrive da capo a ogni lancio |
| `accuracy.json` | l'affidabilità del classificatore sulle immagini vere |

### `accuracy.json`

```json
{
  "epoch": 23,
  "ood_train_images": 868,
  "val":  {"face_color": 0.99, "hair_color": 0.99, "hair": 0.97, "glasses": 0.99, "facial_hair": 0.99, "all": 0.95, "images": 1255},
  "test": {"...": "come val", "images": 1255},
  "ood":  {"...": "come val", "images": 868},
  "ood_pairs": {
    "dark + long":         {"...": "come val", "images": 560},
    "blonde + sunglasses": {"...": "come val", "images": 341}
  }
}
```

I numeri qui sono solo un esempio (anche quelli delle coppie: dipendono da come cade la divisione a metà). Ogni
gruppo ha i 5 attributi, `all` e il numero di immagini. `ood` e `ood_pairs` sono misurati **solo sulla metà di
controllo** delle immagini OOD, quella che il classificatore non ha visto; `ood_train_images` è il numero di
immagini OOD usate nel suo training. I gruppi di `ood_pairs` si prendono dalla colonna `ood_pair` della tabella OOD
(il nome della coppia è quello di `summary.json`, le parole unite da " + "). Le immagini che contengono entrambe le
coppie (66 in tutto lo split OOD) contano in tutti e due i gruppi.

Questi numeri sono il "tetto" della metrica di condizionamento. Se il classificatore riconosce il 99% delle
"dark + long" vere e solo il 70% di quelle generate, gli errori sono quasi tutti del modello di diffusione. Se sulle
vere arrivasse solo al 90%, una parte degli errori sulle generate potrebbe essere del classificatore. In
`evaluate.py` le due accuratezze andranno affiancate.

## Nuove impostazioni in `config.py`

Una sezione "Attribute classifier" con `CLASSIFIER_CHANNELS = (32, 64, 128)`, `CLASSIFIER_EPOCHS = 50` e
`CLASSIFIER_LEARNING_RATE = 1e-3`. Il batch riusa `BATCH_SIZE`.

## Vincoli di stile

Come il resto del progetto: classi e funzioni brevi, docstring con Args e Returns, commenti brevi in inglese che
spiegano il perché, tutte le impostazioni in `config.py`, solo l'essenziale.

## File

```
models/attribute_classifier.py      AttributeClassifier, measure_accuracy
preprocessing/cartoon_dataset.py    attribute_labels(words, config); CartoonDataset.labels
train_classifier.py                 il training, log.csv, classifier.pt, accuracy.json
config.py                           CLASSIFIER_CHANNELS, CLASSIFIER_EPOCHS, CLASSIFIER_LEARNING_RATE
README.md                           una sezione sul classificatore e le righe nella tabella del codice
tests/test_attribute_classifier.py
```

## Test (`tests/test_attribute_classifier.py`, pytest, senza `data/` né GPU)

1. **Forme dell'uscita a 32×32 e a 64×64**: le teste danno (B, 4), (B, 5), (B, 4), (B, 3) e (B, 2). La stessa rete
   funziona alle due risoluzioni.
2. **`attribute_labels`**: "tan, black, long, no glasses, a beard" dà `[1, 3, 3, 2, 0]`. Un ordine sbagliato
   falserebbe tutte le metriche senza nessun errore visibile.
3. **`measure_accuracy`** con un finto classificatore che dà risposte fisse, su 2 immagini: la prima con tutto
   giusto, la seconda con solo i capelli sbagliati. Deve dare `hair` = 0,5, gli altri 4 attributi = 1 e
   `all` = 0,5.

Il training vero si verifica a mano: si lancia `python train_classifier.py` e si controllano `log.csv` e
`accuracy.json`.

## Fuori da questo lavoro

`evaluate.py` (FID, KID, accuratezza sulle immagini generate, diversità tra seed, tempo e memoria) e la funzione
che ricarica `classifier.pt` per usarlo lì. Non sono previste nemmeno le matrici di confusione, l'augmentation (solo
se il log mostra overfitting) e il quarto blocco a 64×64 (solo se l'accuratezza a 64 è bassa).
