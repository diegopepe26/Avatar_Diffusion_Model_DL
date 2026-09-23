"""Step 2: compositional OOD split, then ordinary test, validation and train."""

import random


class DatasetSplitter:
    """Divides the images into train, val, test (ordinary) and ood (held-out pairs)."""

    def __init__(self, config):
        """Store the settings and check that the words of the OOD pairs exist.

        Args:
            config: the project Config (uses OOD_PAIRS, VAL_FRACTION, TEST_FRACTION,
                MIN_IMAGES_TO_SPLIT, SEED, MAPPING, SPLIT_FILES).
        """
        self.config = config
        self.ood_pairs = config.OOD_PAIRS
        for pair in self.ood_pairs:
            for attr, word in pair.items():
                if attr not in config.MAPPING or word not in config.MAPPING[attr]:
                    raise ValueError(f"OOD pair {pair}: '{word}' is not a word of '{attr}' in MAPPING")

    def _pair_name(self, pair):
        """Name of a held-out pair: {'face_color': 'dark', 'hair': 'long'} -> 'dark + long'."""
        return ' + '.join(pair.values())

    def _pairs_in_row(self, row):
        """Find the held-out pairs contained in one image.

        Args:
            row: one row of the captions table (one word per attribute).

        Returns:
            List with the names of the pairs whose words are all in the row (empty if none).
        """
        names = []
        for pair in self.ood_pairs:
            if all(row[attr] == word for attr, word in pair.items()):
                names.append(self._pair_name(pair))
        return names

    def split_ood(self, data):
        """Separate the images that contain a held-out pair.

        Args:
            data: the captions table of all the images.

        Returns:
            (ood, in_distribution): two tables. ood has the extra column 'ood_pair'
            with the names of the pairs it contains.
        """
        pairs = [self._pairs_in_row(row) for _, row in data.iterrows()]
        has_pair = [len(names) > 0 for names in pairs]
        ood = data[has_pair].copy()
        ood['ood_pair'] = ['; '.join(names) for names in pairs if names]
        in_distribution = data[[not flag for flag in has_pair]].copy()
        return ood, in_distribution

    def _split_combination(self, rows, rng):
        """Divide the images of one combination (same caption) between train, val and test.

        Args:
            rows: row indexes of the images with that caption.
            rng: random generator, shared by all the combinations of one split.

        Returns:
            (train_rows, val_rows, test_rows).
        """
        rows = list(rows)
        rng.shuffle(rows)
        if len(rows) < self.config.MIN_IMAGES_TO_SPLIT:
            return rows, [], []   # too few images for train, val and test: all in train
        n_val = max(1, round(len(rows) * self.config.VAL_FRACTION))
        n_test = max(1, round(len(rows) * self.config.TEST_FRACTION))
        return rows[n_val + n_test:], rows[:n_val], rows[n_val:n_val + n_test]

    def split_id(self, in_distribution):
        """Divide the in-distribution images into train, val and ordinary test.

        Each combination is divided on its own, so every combination of val and test
        is also in train, with different images.

        Args:
            in_distribution: second table returned by split_ood.

        Returns:
            (train, val, test) tables.
        """
        # created here and not in __init__: every call gives exactly the same split
        rng = random.Random(self.config.SEED)
        groups = {}   # caption -> row indexes of its images
        for i, caption in zip(in_distribution.index, in_distribution['caption']):
            groups.setdefault(caption, []).append(i)

        train_rows, val_rows, test_rows = [], [], []
        for caption in sorted(groups):   # sorted: same order, same split
            train, val, test = self._split_combination(groups[caption], rng)
            train_rows += train
            val_rows += val
            test_rows += test
        # sorted rows: the tables keep the file order
        return (in_distribution.loc[sorted(train_rows)],
                in_distribution.loc[sorted(val_rows)],
                in_distribution.loc[sorted(test_rows)])

    def check(self, train, val, test, ood):
        """Stop with an error if the split breaks one of its rules.

        Rules: no held-out pair in train, val or test; every word of the pairs is still
        in train (combined with other values); every combination of val and test is in train.

        Args:
            train, val, test, ood: the four tables.
        """
        for name, table in [('train', train), ('val', val), ('test', test)]:
            for _, row in table.iterrows():
                if self._pairs_in_row(row):
                    raise ValueError(f"{row['file']} in {name} contains a held-out pair")

        for pair in self.ood_pairs:
            for attr, word in pair.items():
                if (train[attr] == word).sum() == 0:
                    raise ValueError(f"'{word}' ({attr}) never appears in train")

        train_captions = set(train['caption'])
        for name, table in [('val', val), ('test', test)]:
            unseen = set(table['caption']) - train_captions
            if unseen:
                raise ValueError(f'{len(unseen)} combinations of {name} never appear in train')

    def save(self, train, val, test, ood):
        """Write the four tables to the files in SPLIT_FILES.

        Args:
            train, val, test, ood: the four tables.
        """
        for name, table in [('train', train), ('val', val), ('test', test), ('ood', ood)]:
            table.to_csv(self.config.SPLIT_FILES[name], index=False)

    def summary(self, train, val, test, ood):
        """Collect the numbers the assignment asks to report.

        Args:
            train, val, test, ood: the four tables.

        Returns:
            Dictionary with: images and combinations per split; images of each held-out
            pair; words present in train; what each pair word is combined with in train;
            combinations only in train (too few images to be split).
        """
        summary = {'counts': {}, 'held_out_pairs': {}, 'train_attributes': {},
                   'pair_words_in_train': {}, 'train_only_combinations': {}}

        for name, table in [('train', train), ('val', val), ('test', test), ('ood', ood)]:
            summary['counts'][name] = {'images': len(table), 'combinations': table['caption'].nunique()}

        for pair in self.ood_pairs:
            name = self._pair_name(pair)
            summary['held_out_pairs'][name] = int(ood['ood_pair'].str.contains(name, regex=False).sum())

        for attr, groups in self.config.MAPPING.items():
            summary['train_attributes'][attr] = {word: int((train[attr] == word).sum()) for word in groups}

        # e.g. 'dark (face_color) with hair': {'balding': 101, 'short': 652, 'medium': 311, 'long': 0}
        for pair in self.ood_pairs:
            for attr, word in pair.items():
                rows = train[train[attr] == word]
                for other in pair:
                    if other != attr:
                        summary['pair_words_in_train'][f'{word} ({attr}) with {other}'] = {
                            w: int((rows[other] == w).sum()) for w in self.config.MAPPING[other]}

        evaluated = set(val['caption']) | set(test['caption'])
        for caption in sorted(set(train['caption']) - evaluated):
            summary['train_only_combinations'][caption] = int((train['caption'] == caption).sum())
        return summary
