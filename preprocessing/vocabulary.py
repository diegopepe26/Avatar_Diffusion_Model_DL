"""Step 3: word-level vocabulary, built only from the training captions."""

import json
import re


class Vocabulary:
    """Maps the tokens of a caption to integer ids and back.

    Built once by the pipeline (build + save), then loaded by the training and the
    generation (load), so everybody uses the same ids.
    """

    def __init__(self, config):
        """Start with the special tokens only.

        Args:
            config: the project Config (uses SPECIAL_TOKENS: '<pad>', '<unk>', '<bos>', '<eos>').
        """
        self.tokens = list(config.SPECIAL_TOKENS)   # the position of a token is its id
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.max_length = 0

    def tokenize(self, text):
        """Split a text into lowercase tokens: words and single punctuation marks.

        Args:
            text: caption or prompt, e.g. 'Pale skin, short hair'.

        Returns:
            List of tokens, e.g. ['pale', 'skin', ',', 'short', 'hair'].
        """
        # lowercase: in a hand-written prompt 'Blonde' and 'blonde' are the same token
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def build(self, captions):
        """Add the words of the training captions, from the most frequent.

        Same frequency: alphabetical order, so the ids never change between runs.

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

    def check_captions(self, captions):
        """Stop with an error if a caption has a word the vocabulary does not know,
        or does not fit in max_length.

        Args:
            captions: the captions of val, test and ood.
        """
        for caption in captions:
            for token in self.tokenize(caption):
                if token not in self.ids:
                    raise ValueError(f"'{token}' is not in the training vocabulary: {caption}")
            self.encode(caption)   # stops if the caption is too long

    def encode(self, text):
        """Convert a caption into max_length ids: <bos>, tokens, <eos>, then <pad>.

        Args:
            text: caption or prompt; unknown words become <unk>.

        Returns:
            List of max_length integers, the input of the text encoder.
        """
        ids = [self.ids['<bos>']]
        ids += [self.ids.get(token, self.ids['<unk>']) for token in self.tokenize(text)]
        ids += [self.ids['<eos>']]
        if len(ids) > self.max_length:
            raise ValueError(f'too long: {len(ids)} tokens, the maximum is {self.max_length}: {text}')
        return ids + [self.ids['<pad>']] * (self.max_length - len(ids))

    def decode(self, ids):
        """Convert ids back into text, to check what the model receives.

        Args:
            ids: list (or tensor) of ids.

        Returns:
            The text without <pad>, <bos> and <eos>.
        """
        tokens = [self.tokens[int(i)] for i in ids]
        tokens = [t for t in tokens if t not in ('<pad>', '<bos>', '<eos>')]
        return ' '.join(tokens).replace(' ,', ',')   # 'skin , short' -> 'skin, short'

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
