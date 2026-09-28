# Valutazione di un esperimento (`evaluate.py`): design

Data: 2026-09-28 · Branch: `evaluate` (da `demo-classifier`)

## Obiettivo

Uno script che misura un esperimento già addestrato: il modello condizionato, la baseline senza testo e, più
avanti, quelli a 64×64. Lo misura sulle caption del test ordinario e su quelle OOD, e calcola tutte le metriche che
la traccia chiede come obbligatorie ("Mandatory metrics"):

- una metrica di **qualità**: FID e KID, con tutti i dettagli di come sono calcolati;
- la **diversità tra seed** per lo stesso prompt;
- il **numero di parametri**, il **tempo di generazione** e la **memoria** usata.

A queste si aggiunge la metrica di **condizionamento**, fatta con il classificatore degli attributi. Serve a
rispondere alla domanda di ricerca: il modello rispetta il prompt anche sulle combinazioni mai viste nel training
(OOD)?

Tutto sta in uno script solo perché la parte che costa è generare le immagini (circa 75-80 minuti a 32×32),
mentre le metriche richiedono pochi minuti: si genera una volta e si calcola tutto sulle stesse immagini.

## Decisioni prese nel brainstorming

- **Guidance fissa a 3** (`GUIDANCE_SCALE`). Non si confrontano valori diversi di guidance: la traccia non lo
  chiede, e ogni valore costerebbe altri 75-80 minuti di generazione.
- **Un'immagine per ogni riga** del test (1.255) e delle OOD (1.736), con la caption di quella riga: 2.991
  immagini. Così ogni gruppo di immagini generate ha la stessa dimensione del gruppo di immagini vere con cui si
  confronta, e le caption compaiono nelle stesse proporzioni. Con un'immagine per combinazione (444 in tutto) il
  FID sarebbe poco affidabile.
- **Diversità misurata con la differenza dei pixel.** La distanza tra due immagini è la media, sui 3.072 valori
  (32 × 32 pixel × 3 colori), della differenza senza segno, con i pixel nella scala da 0 a 1. Non richiede reti ed
  è facile da spiegare. Gli avatar sono tutti centrati nella stessa posizione, quindi lo stesso pixel è sempre la
  stessa parte della faccia.
- **La diversità si misura sugli 8 `SAMPLE_PROMPTS`** di `config.py` (4 combinazioni viste nel training, 4 OOD),
  con 16 immagini ciascuno, generate in un solo gruppo. Il riferimento sono tutte le immagini vere del dataset con
  la stessa caption, da 9 a 47 secondo la caption.
- **Un seed per ogni gruppo, non per ogni immagine.** La demo continua a generare un'immagine alla volta, con un
  seed per immagine, così "seed 7" identifica sempre la stessa immagine. La valutazione genera in gruppo, circa
  1,5 secondi per immagine invece di 19. I due modi usano lo stesso modello, la stessa funzione `sample` e la stessa
  guidance: cambia solo come i numeri casuali vengono distribuiti tra le immagini. Le immagini della valutazione
  sono riproducibili rilanciando lo script, ma non si possono ricreare una per una nella demo. Un generatore per
  immagine anche in gruppo è possibile modificando `sample` e `step`, ma resta fuori da questo lavoro.
- **FID e KID di riferimento**, calcolati tra due gruppi di immagini vere: danno il valore migliore che si può
  sperare di ottenere con quel numero di immagini.
- **La baseline passa per lo stesso script**, con le stesse metriche, senza casi speciali. Le caption servono solo
  a dire quante immagini generare, perché la baseline non ha il text encoder e i token non vengono mai letti. Il
  condizionamento della baseline è il livello "per fortuna", FID e KID sono la qualità senza testo, e la diversità è
  la varietà degli avatar senza nessun vincolo.
- **Griglie delle immagini "sbagliate"**, con un file di testo accanto, per controllare a occhio se un errore è del
  modello o del classificatore.

## Come si usa e cosa produce

```
python evaluate.py runs/conditional_32
```

La cartella è quella dell'esperimento, come per `make_training_gif.py`. Le impostazioni della rete si leggono dal
checkpoint con `load_model`, quindi lo script funziona per qualunque esperimento. Il classificatore è quello della
risoluzione dell'esperimento, `runs/classifier_<IMAGE_SIZE>`, caricato con `load_classifier`.

Tutti i file vanno in `runs/<esperimento>/evaluation/`:

| File | Contenuto |
|---|---|
| `evaluation.json` | tutti i numeri (struttura più sotto) |
| `generated_test.pt`, `generated_ood.pt` | le 1.255 e 1.736 immagini generate, più i secondi e la memoria della loro generazione |
| `generated_diversity.pt` | le 128 immagini della diversità, più secondi e memoria |
| `diversity.png` | la griglia della diversità |
| `disagreements_test.png`, `disagreements_ood.png` | le prime 32 immagini generate con almeno una parola sbagliata |
| `disagreements_test.txt`, `disagreements_ood.txt` | una riga per ogni immagine della griglia: la caption e le parole che non tornano |

Le immagini generate si salvano come interi da 0 a 255, come i PNG delle immagini vere: il classificatore e FID/KID
vedono le generate nello stesso formato delle vere. Occupano circa 4 MB (test) e 5 MB (OOD) a 32×32. Se un file
`generated_*.pt` esiste già, lo script lo legge invece di rigenerare. Così, se si blocca dopo la generazione (per
esempio durante il FID) o se si corregge una metrica, rilanciarlo richiede pochi minuti. Per ricominciare da capo,
per esempio dopo un nuovo training del modello, si cancella la cartella `evaluation/`.

## I controlli iniziali

Prima di generare, lo script si ferma con un `ValueError` e un messaggio che dice cosa fare se:

- nella cartella non c'è `last.pt` ("train it first with train.py");
- non c'è il classificatore della risoluzione dell'esperimento ("run train_classifier.py with IMAGE_SIZE = N");
- l'epoca scritta in `accuracy.json` è diversa da quella salvata in `classifier.pt`. Succede se un training del
  classificatore viene interrotto: `classifier.pt` viene sovrascritto già alla prima epoca, mentre `accuracy.json`
  si scrive solo alla fine. Useremmo un giudice con il voto di affidabilità di un altro ("run train_classifier.py
  again").

Senza GPU stampa un avviso, come `train.py`: sulla CPU la generazione durerebbe molte ore.

## La generazione

- **Ordine.** Le immagini si generano nell'ordine delle righe dello split: l'immagine k nasce dalla caption della
  riga k, e dopo si confronta con le etichette della riga k.
- **Gruppi.** Si genera con `sample` a gruppi di `BATCH_SIZE` (128) immagini. Test: 9 gruppi da 128 e uno da 103
  (10 gruppi). OOD: 13 gruppi da 128 e uno da 72 (14 gruppi). Diversità: un gruppo da 128 (gli 8 prompt ripetuti
  16 volte, ognuno 16 volte di seguito).
- **Seed.** Ogni gruppo ha il suo seed, numerati di seguito a partire da `SEED` (42): i gruppi del test usano da 42
  a 51, quelli delle OOD da 52 a 65, quello della diversità 66. Se tutti i gruppi usassero lo stesso seed, le
  immagini nella stessa posizione di gruppi diversi partirebbero dallo stesso rumore e riceverebbero lo stesso
  rumore nuovo a ogni passo: verrebbero molto simili in tutto quello che la caption non dice, e il gruppo delle
  generate sarebbe meno vario di quanto il modello sa fare.
- **Guidance.** `sample` usa `GUIDANCE_SCALE` del config dell'esperimento, cioè 3. Non si usa `generate` di
  `generate.py`, che cambia per sempre la guidance nel config che riceve.

## Parametri, tempo e memoria

- **Parametri**: in totale, nel text encoder (0 per la baseline) e nella U-Net.
- **Tempo in gruppo**: i secondi della generazione del test più quelli delle OOD, divisi per le 2.991 immagini. I
  secondi si salvano nei file `generated_*.pt`, così restano quelli veri anche quando le immagini vengono rilette.
- **Tempo di un'immagine da sola**: si genera una sola immagine (la prima caption del test, seed `SEED`) e si
  misura, a ogni lancio. È il caso della demo: circa 19 secondi per il modello condizionato, circa 10 per la
  baseline, che a ogni passo usa il modello una volta sola.
- **Memoria**: prima di generare si azzera il conto della memoria massima della GPU
  (`torch.cuda.reset_peak_memory_stats`), dopo si legge il valore più alto raggiunto
  (`torch.cuda.max_memory_allocated`), in MB. Si misura per la generazione in gruppo (il più alto tra quello del
  test e quello delle OOD, salvati nei file `generated_*.pt`) e per l'immagine da sola. Senza GPU il valore è
  `null`.

## Il condizionamento

`measure_accuracy` riceve le immagini generate (riportate da 0-255 a [-1, 1]) e le etichette delle righe dello
split, e restituisce la percentuale di immagini giuste per ogni attributo e con tutte e cinque le parole giuste
(`all`). Si calcola:

- sul test (1.255 immagini);
- sulle OOD (tutte le 1.736: il modello di diffusione non ha mai visto nessuna combinazione OOD, quindi non serve la
  divisione a metà che serviva al classificatore);
- separatamente sulle due coppie tenute fuori, scegliendo le righe con la colonna `ood_pair` (con `in`, come in
  `train_classifier.py`: un'immagine con tutte e due le coppie conta in tutti e due i gruppi).

Accanto a ogni risultato si scrive quello del classificatore sulle immagini vere, preso da `accuracy.json` (test,
metà di controllo delle OOD, coppie), insieme all'epoca e all'accuratezza su val del classificatore.

## La diversità

Per ogni `SAMPLE_PROMPT`:

1. le 16 immagini generate formano 16 × 15 / 2 = 120 coppie; per ogni coppia si calcola la distanza (media dei
   3.072 valori della differenza senza segno, pixel tra 0 e 1);
2. la media delle 120 distanze è la diversità del prompt;
3. lo stesso calcolo sulle immagini vere con la stessa caption, prese da tutti e quattro gli split (train, val, test,
   OOD), dà il riferimento: con n immagini vere le coppie sono n × (n − 1) / 2;
4. un prompt è OOD se la sua caption compare nella tabella delle OOD.

Poi si fa la media delle diversità dei 4 prompt visti e, separatamente, dei 4 prompt OOD, sia per le generate sia
per le vere. Le medie di prompt con un numero diverso di immagini si possono confrontare, perché ognuna stima la
stessa cosa: la distanza tipica tra due avatar con quella caption. Il riferimento di un prompt con poche immagini
vere è solo meno preciso, per questo si scrive anche il numero di immagini vere.

`diversity.png` è disegnata con `save_preview` (8 colonne, immagini ingrandite 4 volte): le 16 immagini di un
prompt occupano due righe, quindi le righe 1-2 sono il primo prompt, le righe 3-4 il secondo, e così via.

## FID e KID

Si usa `torchmetrics` (`FrechetInceptionDistance` e `KernelInceptionDistance`), che si appoggia a `torch-fidelity`,
l'implementazione di riferimento. Nuova dipendenza in `requirements.txt`: `torchmetrics[image]`.

- **Inception-v3**, preaddestrata su ImageNet (pesi `pt_inception-2015-12-05`, scaricati da internet la prima volta,
  circa 100 MB). È ammessa perché serve solo a valutare. Di ogni immagine si usano i 2.048 numeri dello strato
  prima della risposta finale.
- Le immagini si passano con valori tra 0 e 1 (`normalize=True`); la libreria le porta a 299×299 con un
  ingrandimento bilineare, la dimensione che Inception vuole.
- **FID**: per ogni gruppo si calcolano la media dei 2.048 numeri e la loro dispersione (matrice di covarianza
  2.048 × 2.048); FID = ‖media₁ − media₂‖² + Tr(Σ₁ + Σ₂ − 2·(Σ₁·Σ₂)^½). Vale 0 per gruppi uguali.
- **KID**: confronta le immagini a due a due con una funzione di somiglianza (un kernel polinomiale) sugli stessi
  2.048 numeri; non deve stimare la matrice 2.048 × 2.048 e in media non viene gonfiato quando le immagini sono
  poche. Si calcola su 100 sottogruppi di 500 immagini presi a caso da ogni gruppo, e si riportano la media e la
  deviazione standard dei 100 valori. 500 perché il gruppo più piccolo ha 868 immagini e un sottogruppo non può
  essere più grande (il valore di default, 1.000, darebbe errore). Prima del KID si fissa il seed, così anche i
  sottogruppi sono riproducibili. I valori si scrivono così come sono (nel report si possono moltiplicare per
  1.000, come si fa spesso).

I quattro confronti:

| Confronto | Immagini | Cosa dice |
|---|---|---|
| generate da test contro vere di test | 1.255 e 1.255 | qualità sulle combinazioni viste |
| generate da OOD contro vere OOD | 1.736 e 1.736 | qualità sulle combinazioni mai viste |
| vere di val contro vere di test | 1.255 e 1.255 | riferimento per il test |
| metà delle vere OOD contro l'altra metà | 868 e 868 | riferimento per le OOD |

Le due metà delle OOD vere si scelgono mescolando le righe con un generatore inizializzato con `SEED`. Il FID dipende
dal numero di immagini: il FID del test e quello delle OOD non sono perfettamente confrontabili, e il riferimento
OOD, con 868 immagini per gruppo, è un po' gonfiato. Per confrontare test e OOD è più affidabile il KID. Il calcolo
si fa sulla GPU e dura qualche minuto.

## Le griglie delle immagini "sbagliate"

Per il test e per le OOD: le prime 32 immagini generate, nell'ordine delle righe, su cui il classificatore trova
almeno una parola diversa dalla caption. Si disegnano con `save_preview` (4 righe da 8, ingrandite 4 volte). Nel
file `.txt`, una riga per immagine nello stesso ordine della griglia (da sinistra a destra, dall'alto in basso):

```
5. a cartoon avatar with dark skin, long black hair, glasses and a beard — hair: asked long, sees medium
```

Gli attributi sbagliati sono elencati nell'ordine di `MAPPING`. Se le immagini sbagliate sono meno di 32 la griglia
ne ha meno; se non ce n'è nessuna, il file di testo lo dice e la griglia non si crea.

## `evaluation.json`

```json
{
  "experiment": "conditional_32",
  "image_size": 32,
  "text_conditioning": true,
  "guidance": 3.0,
  "parameters": {"total": 6145731, "text_encoder": 400256, "unet": 5745475},
  "sampling": {
    "device": "cuda",
    "batch_size": 128,
    "seconds_per_image_in_batch": 1.5,
    "peak_memory_mb_batch": 1200.0,
    "seconds_single_image": 19.0,
    "peak_memory_mb_single": 150.0
  },
  "conditioning": {
    "judge": {"epoch": 40, "val_accuracy": 0.9865},
    "test": {"generated": {"face_color": 0.99, "...": "...", "all": 0.95, "images": 1255},
             "real": {"...": "da accuracy.json"}},
    "ood": {"generated": {"...": "...", "images": 1736}, "real": {"...": "..."}},
    "ood_pairs": {"dark + long": {"generated": {"...": "..."}, "real": {"...": "..."}},
                  "blonde + sunglasses": {"generated": {"...": "..."}, "real": {"...": "..."}}}
  },
  "quality": {
    "settings": {"library": "torchmetrics (torch-fidelity)", "network": "Inception-v3, pt_inception-2015-12-05",
                 "features": 2048, "resize": "299x299 bilinear", "kid_subsets": 100, "kid_subset_size": 500},
    "test": {"fid": 25.0, "kid_mean": 0.004, "kid_std": 0.001, "images": [1255, 1255]},
    "ood": {"...": "...", "images": [1736, 1736]},
    "reference_test": {"...": "vere di val contro vere di test", "images": [1255, 1255]},
    "reference_ood": {"...": "metà contro metà delle vere OOD", "images": [868, 868]}
  },
  "diversity": {
    "images_per_prompt": 16,
    "distance": "mean absolute pixel difference, pixels in [0, 1]",
    "prompts": [{"prompt": "a cartoon avatar with ...", "held_out": false, "generated": 0.12, "real": 0.15,
                 "real_images": 27}],
    "seen": {"generated": 0.12, "real": 0.15},
    "held_out": {"generated": 0.11, "real": 0.14}
  }
}
```

I numeri qui sono solo un esempio della forma (tranne i parametri, che sono quelli veri di `conditional_32`).

## Nuove impostazioni in `config.py`

Una sezione "Evaluation":

```python
DIVERSITY_IMAGES = 16       # images per SAMPLE_PROMPT for the diversity across seeds
KID_SUBSETS = 100           # random subsets whose KIDs are averaged
KID_SUBSET_SIZE = 500       # images in every subset: the smallest group (half of the real OOD) has 868
DISAGREEMENTS_SHOWN = 32    # generated images with a wrong word shown in the grids
```

## Vincoli di stile

Come il resto del progetto: funzioni brevi, docstring con Args e Returns in cui ogni lettera di una forma è
spiegata (per esempio "N = number of images"), commenti brevi in inglese che spiegano il perché, tutte le
impostazioni in `config.py`, `ValueError` con un messaggio chiaro, solo l'essenziale. `evaluate.py` sta nella
cartella principale, come `train.py`: una funzione per ogni cosa da calcolare e un `main` che le chiama in ordine.
Riusa `load_model`, `load_classifier`, `measure_accuracy`, `sample`, `CartoonDataset` e
`ImagePreprocessor.save_preview`.

## File

```
evaluate.py                 la valutazione
config.py                   sezione "Evaluation"
requirements.txt            + torchmetrics[image]
README.md                   sezione "Evaluation" e riga nella tabella del codice
tests/test_evaluate.py
```

## Test (`tests/test_evaluate.py`, pytest, senza `data/` né GPU)

1. **Il conto della diversità**: tre immagini finte, due tutte nere (−1) e una tutta bianca (1). Le coppie sono
   tre (nera-nera 0, nera-bianca 1, nera-bianca 1) e la media deve essere 2/3.
2. **I seed dei gruppi**: con un modello piccolo non addestrato (`NUM_TIMESTEPS = 10`), 3 immagini con la stessa
   caption a gruppi da 2. La prima immagine del secondo gruppo deve essere diversa dalla prima del primo gruppo, e
   rifacendo la stessa generazione le immagini devono venire identiche.
3. **Il riuso delle immagini salvate**: se il file `generated_*.pt` esiste, le immagini si leggono da lì; al posto
   del modello si passa qualcosa che non può generare, così una rigenerazione farebbe fallire il test.
4. **Il controllo del giudice**: un `classifier.pt` salvato con l'epoca 40 e un `accuracy.json` con l'epoca 30 devono
   dare un `ValueError`.
5. **La riga del file degli errori**: data una caption, le parole chieste e quelle viste, la riga elenca solo gli
   attributi sbagliati, nell'ordine di `MAPPING`, con la parola chiesta e quella vista.

FID e KID non hanno un test automatico: il calcolo è della libreria, e controllarlo richiederebbe i pesi di Inception
e immagini vere.

## Verifica finale

Si lancia `python evaluate.py runs/conditional_32` (circa 80 minuti) e si controlla che:

- tutti i file di `evaluation/` vengano creati;
- i riferimenti "vere contro vere" abbiano FID e KID più bassi dei confronti con le generate;
- il condizionamento sul test sia vicino a quello misurato dal revisore (100% su 251 prompt di test con guidance 3);
- la diversità delle generate sia maggiore di zero e confrontabile con quella delle vere;
- un secondo lancio riusi le immagini salvate e finisca in pochi minuti.

La baseline non è ancora addestrata: la sua valutazione si farà quando esisterà `runs/unconditional_32`.

## Fuori da questo lavoro

Il confronto tra valori di guidance, un generatore per immagine anche nella generazione in gruppo (modifica di
`sample` e `step`), LPIPS, il training della baseline e quello a 64×64, le analisi degli errori oltre alle griglie
(per esempio quali parole vengono confuse con quali): si possono fare dopo, rileggendo le immagini salvate.
