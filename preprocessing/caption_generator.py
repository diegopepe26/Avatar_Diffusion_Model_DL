"""Step 1: from the attribute files of the Cartoon Set to one caption per image."""

import csv
import os

import pandas as pd


class CaptionGenerator:
    """Turns the attribute values of every image into words and a caption."""

    def __init__(self, config):
        """Args:
            config: the project Config (uses MAPPING, TEMPLATE, the raw folders, CAPTIONS_FILE).
        """
        self.config = config
        self.mapping = config.MAPPING
        self.template = config.TEMPLATE
        self.lookup = self._invert_mapping()

    def _invert_mapping(self):
        """Turn MAPPING (attribute -> word -> values) into attribute -> value -> word.

        Returns:
            The lookup table, e.g. lookup['hair'][54] == 'short'.
        """
        lookup = {}
        for attr, groups in self.mapping.items():
            lookup[attr] = {}
            for word, values in groups.items():
                for value in values:
                    if value in lookup[attr]:
                        raise ValueError(f"{attr}: value {value} is both in '{lookup[attr][value]}' and in '{word}'")
                    lookup[attr][value] = word
        return lookup

    def _read_attributes(self, csv_path):
        """Read the attribute file of one image.

        Args:
            csv_path: file with lines like '"hair", 54, 111' (name, value, number of values).

        Returns:
            {attribute: value} for all the attributes in the file.
        """
        with open(csv_path, newline='') as f:
            # skipinitialspace removes the space after the comma: ' 54' becomes '54'
            return {row[0]: int(row[1]) for row in csv.reader(f, skipinitialspace=True) if row}

    def compose_caption(self, words):
        """Fill the template. Public: also used at generation time to write the prompts.

        Args:
            words: {attribute: word}, e.g. {'face_color': 'pale', 'hair': 'long', ...}.

        Returns:
            The caption, e.g. 'a cartoon avatar with pale skin, long blonde hair, ...'.
        """
        return self.template.format(**words)

    def generate_dataset(self):
        """Build one row per image and save them to CAPTIONS_FILE.

        Returns:
            DataFrame with the columns file, <attr> and <attr>_value for each attribute, caption.
        """
        rows = []
        missing = 0
        # sorted: same order every time, so the same seed gives the same split
        for name in sorted(os.listdir(self.config.ATTRIBUTES_DIR)):
            if not name.lower().endswith('.csv'):
                continue
            png = os.path.splitext(name)[0] + '.png'
            if not (self.config.IMAGES_DIR / png).is_file():
                missing += 1
                continue

            values = self._read_attributes(self.config.ATTRIBUTES_DIR / name)
            row = {'file': png}
            words = {}
            for attr in self.mapping:
                if values[attr] not in self.lookup[attr]:
                    raise ValueError(f'{name}: value {values[attr]} of {attr} has no word in MAPPING')
                words[attr] = self.lookup[attr][values[attr]]
                row[attr] = words[attr]               # the word, e.g. 'short'
                row[attr + '_value'] = values[attr]   # the original number, e.g. 54
            row['caption'] = self.compose_caption(words)
            rows.append(row)

        if missing:
            print(f'Warning: {missing} images have no PNG file and were skipped')
        data = pd.DataFrame(rows)
        data.to_csv(self.config.CAPTIONS_FILE, index=False)
        return data
