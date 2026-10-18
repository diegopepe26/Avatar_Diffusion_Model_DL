"""Step 3 of the data preparation, a word-level vocabulary built only from the training captions."""

import json
import re


class Vocabulary:
    """Maps the tokens of a caption to integer ids and back.

    Built once by the pipeline, then loaded by the training and the generation, so everybody uses the same ids.
    """

    def __init__(self, config):
        """Start with the special tokens only.

        Args:
            config: the project Config (uses SPECIAL_TOKENS, i.e. '<pad>', '<unk>', '<bos>', '<eos>').
        """
        self.tokens = list(config.SPECIAL_TOKENS)   # the position of a token is its id
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.max_length = 0

    def tokenize(self, text):
        """Split a text into lowercase tokens, which are words and single punctuation marks.

        Args:
            text: caption or prompt, e.g. 'Pale skin, short hair'.

        Returns:
            List of tokens, e.g. ['pale', 'skin', ',', 'short', 'hair'].
        """
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def build(self, captions):
        """Add the words of the training captions, from the most frequent.

        Words with the same frequency go in alphabetical order, so the ids never change between runs.

        Args:
            captions: the training captions (never validation, test or ood ones).
        """
        counts = {}
        for caption in captions:
            for token in self.tokenize(caption):
                counts[token] = counts.get(token, 0) + 1

        for word in sorted(counts, key=lambda word: (-counts[word], word)):
            self.ids[word] = len(self.tokens)
            self.tokens.append(word)

        self.max_length = max(len(self.tokenize(c)) for c in captions) + 2   # + <bos> and <eos>

    def encode(self, text):
        """Convert a caption into ids (<bos>, the tokens, <eos>, then <pad> up to max_length).

        Args:
            text: caption or prompt; unknown words become <unk>.

        Returns:
            List of integers, the input of the text encoder. It has max_length ids for every caption of the template,
            while a text with more than max_length - 2 tokens gives one id per token plus <bos> and <eos>.
        """
        ids = [self.ids['<bos>']]
        ids += [self.ids.get(token, self.ids['<unk>']) for token in self.tokenize(text)]
        ids += [self.ids['<eos>']]

        return ids + [self.ids['<pad>']] * (self.max_length - len(ids))

    def save(self, path):
        """Write the vocabulary to a JSON file.

        Args:
            path: destination file (VOCABULARY_FILE).
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'max_length': self.max_length, 'tokens': self.ids}, f, indent=2)

    def load(self, path):
        """Read a vocabulary written by save (replaces the current tokens).

        Args:
            path: JSON file (VOCABULARY_FILE).
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        self.ids = data['tokens']
        self.tokens = sorted(self.ids, key=self.ids.get)   # back to id order
        self.max_length = data['max_length']
