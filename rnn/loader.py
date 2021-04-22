from collections import namedtuple
from typing import Tuple, Generator

import numpy as np

Tokens = namedtuple("Tokens", ["int_to_chars", "chars_to_int", "tokens"])


class DataLoader:
    def __init__(self, file_path: str):
        self._text = self._read_text_file(file_path)
        self._tokens = self._tokenize(self._text)

    @property
    def tokens(self) -> Tokens:
        return self._tokens

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        with open(file_path, 'r') as f:
            text = f.read()
        return text

    @staticmethod
    def _tokenize(text: str) -> Tokens:
        chars = set(text)
        int_to_chars = dict(enumerate(chars))
        chars_to_int = {char: i for i, char in int_to_chars.items()}
        tokens = np.array([chars_to_int[char] for char in text])
        return Tokens(int_to_chars=int_to_chars, chars_to_int=chars_to_int,
                      tokens=tokens)

    @staticmethod
    def one_hot_encode(tokens: np.array, label_counts: int) -> np.array:
        result = np.zeros((tokens.size, label_counts), dtype=np.float32)
        result[np.arange(result.shape[0]), tokens.flatten()] = 1
        result = result.reshape((*tokens.shape, label_counts))
        return result

    @staticmethod
    def generate_batches(
            sequence: np.array, batch_size: int,
            window: int) -> Generator[Tuple[np.array, np.array], None, None]:
        batch_length = batch_size * window
        batch_count = len(sequence) // batch_length

        truncated_size = batch_count * batch_length
        _sequence = sequence[:truncated_size]
        _sequence = _sequence.reshape((batch_size, -1))

        for n in range(0, _sequence.shape[1], window):
            x = _sequence[:, n:n + window]
            y = np.zeros_like(x)
            if n + window < _sequence.shape[1]:
                y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, n + window]
            else:
                y[:, :-1], y[:, -1] = x[:, 1:], _sequence[:, 0]
            yield x, y
