"""Trains the attribute classifier, the judge of the conditioning metric, on the real training images and on
half of the real OOD images (the diffusion model never sees them: the classifier is only a measuring tool).

Run it with:  python train_classifier.py    (after prepare_data.py; the size is IMAGE_SIZE in config.py)
The results go to runs/classifier_<IMAGE_SIZE>/: classifier.pt (the best epoch on val), log.csv, and
accuracy.json (how often it is right on the real val and test images and on the other half of the OOD images).
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
    """Train for CLASSIFIER_EPOCHS epochs, keep the best one on val, then measure it on val, test and OOD check half.

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
    labels = {split: dataset.labels for split, dataset in data.items()}
    # Trained on train only, the classifier took shortcuts on the held-out pairs (e.g. dark + long -> tan skin):
    # half of the real OOD images, chosen with SEED, also trains it, so it knows every combination;
    # the other half (the check half) measures how often it is right on them.
    order = torch.randperm(len(labels['ood']), generator=torch.Generator().manual_seed(config.SEED))
    ood_train, ood_check = order[:len(order) // 2], order[len(order) // 2:]
    train_images = torch.cat([images['train'], images['ood'][ood_train]])
    train_labels = torch.cat([labels['train'], labels['ood'][ood_train]])
    images['ood'], labels['ood'] = images['ood'][ood_check], labels['ood'][ood_check]   # from here on: the check half
    pair_names = data['ood'].table['ood_pair'].to_numpy()[ood_check.numpy()]           # e.g. 'dark + long'
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=config.BATCH_SIZE, shuffle=True)
    print(f'2. Data: train {len(labels["train"])} + {len(ood_train)} OOD images, val {len(labels["val"])}, '
          f'test {len(labels["test"])}, OOD check half {len(ood_check)}')

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
        for batch, batch_labels in train_loader:
            loss = classifier_loss(classifier(batch.to(device)), batch_labels.to(device), config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        # both in eval(), so they can be compared: a growing gap means overfitting
        train_accuracy = measure_accuracy(classifier, train_images, train_labels, config)['all']
        val_accuracy = measure_accuracy(classifier, images['val'], labels['val'], config)['all']
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

    # 5. The best epoch on the real images it has never seen: val, test, the OOD check half and each held-out pair
    checkpoint = torch.load(checkpoint_file, map_location=device)
    classifier.load_state_dict(checkpoint['weights'])
    report = {'epoch': checkpoint['epoch'], 'ood_train_images': len(ood_train)}
    for split in ['val', 'test', 'ood']:
        report[split] = measure_accuracy(classifier, images[split], labels[split], config)
        report[split]['images'] = len(labels[split])
    report['ood_pairs'] = {}
    for pair in config.OOD_PAIRS:
        name = ' + '.join(pair.values())     # as in the column ood_pair, e.g. 'dark + long'
        # 'in', not ==: an image with both pairs has 'dark + long; blonde + sunglasses' and counts in both
        rows = torch.tensor([name in names for names in pair_names])
        report['ood_pairs'][name] = measure_accuracy(classifier, images['ood'][rows], labels['ood'][rows], config)
        report['ood_pairs'][name]['images'] = int(rows.sum())
    with open(run_dir / 'accuracy.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'5. Best epoch {checkpoint["epoch"]}: all five words right in {report["test"]["all"]:.1%} of test, '
          f'{report["ood"]["all"]:.1%} of the OOD check half (accuracy.json)')
    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main(Config())
