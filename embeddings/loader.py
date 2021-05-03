import typing
from abc import ABCMeta, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np
import nltk
import torch
from nltk.corpus import stopwords


@dataclass
class TextDataSummary:
    unique_words: typing.List[str]
    word_tokens: typing.Dict[str, int]
    word_counts: typing.Counter
    word_probability: typing.Dict[str, float]


class DataLoader(metaclass=ABCMeta):
    punctuations = {
        '.': ' <PERIOD> ',
        ',': ' <COMMA> ',
        '"': ' <QUOTATION_MARK> ',
        '\'': ' <QUOTATION_MARK> ',
        ';': ' <SEMICOLON> ',
        '!': ' <EXCLAMATION_MARK> ',
        '?': ' <QUESTION_MARK> ',
        '(': ' <LEFT_PAREN> ',
        ')': ' <RIGHT_PAREN> ',
        '--': ' <HYPHENS> ',
        ':': ' <COLON> '
    }

    def __init__(self, file_path: str, remove_frequency: int,
                 batch_size: int, context_window: int, device: str = "cpu"):
        self._batch_size = batch_size
        self._context_window = context_window
        self._device = device

        with open(file_path) as f:
            self._text = f.read()
        words = self._pre_process(text=self._text,
                                  remove_frequency=remove_frequency)
        batch_count = len(words) // batch_size
        self._words = words[:(batch_size * batch_count)]
        self._summary = self._calculate_summary(words=self._words)

    @property
    def text(self):
        return self._text

    @property
    def words(self):
        return self._words

    @property
    def summary(self):
        return self._summary

    def _pre_process(self, text, remove_frequency):
        nltk.download("stopwords")

        # Replace punctuation markers
        text = text.lower()
        for key, value in self.punctuations.items():
            text = text.replace(key, value)
        words = text.split()

        stop_words_set = set(stopwords.words())
        for punctuation_token in self.punctuations.values():
            stop_words_set.add(punctuation_token.strip())

        def exceeded_count(w: str) -> bool:
            return word_counts[w] > remove_frequency

        def is_stop_word(w: str) -> bool:
            return w in stop_words_set

        word_counts = Counter(words)
        filtered = [word for word in words if
                    (exceeded_count(word) and (not is_stop_word(word)))]
        return filtered

    @staticmethod
    def _calculate_summary(words: typing.List[str]) -> TextDataSummary:
        word_counts = Counter(words)
        unique_words = list(word_counts.keys())
        unique_words.sort(key=lambda x: word_counts[x], reverse=True)
        word_tokens = {w: i for i, w in enumerate(unique_words)}
        total_counts = sum(word_counts.values())
        word_prob = {w: c / total_counts for w, c in word_counts.items()}
        return TextDataSummary(unique_words=unique_words,
                               word_tokens=word_tokens,
                               word_counts=word_counts,
                               word_probability=word_prob)

    def _get_context_tokens(self, i: int, window: int) -> typing.List[int]:
        half = window // 2
        left = max(0, i - half)
        right = min(left + window, len(self._words))
        if right == len(self._words):
            left = right - window
        surrounding_words = self._words[left:right]
        context_tokens = [self._summary.word_tokens[x] for x in
                          surrounding_words if x != self._words[i]]
        return context_tokens

    @abstractmethod
    def get_batches(self) -> typing.Tuple:
        raise NotImplementedError()


class SGLoader(DataLoader):

    def get_batches(self) -> typing.Tuple[torch.LongTensor,
                                          torch.LongTensor]:
        for idx in range(0, len(self._words), self._batch_size):
            target, context = [], []
            batch = self._words[idx:idx + self._batch_size]
            for i, word in enumerate(batch):
                word_token = self.summary.word_tokens[word]
                context_tokens = self._get_context_tokens(i,
                                                          self._context_window)
                target.extend([word_token] * len(context_tokens))
                context.extend(context_tokens)
            yield torch.LongTensor(target).to(self._device), torch.LongTensor(
                context).to(self._device)


class SGNSLoader(DataLoader):
    UNIGRAM_FACTOR = 0.75

    def __init__(self, file_path: str, remove_frequency: int, batch_size: int,
                 context_window: int, noise_count: int, **kwargs):
        super().__init__(file_path, remove_frequency, batch_size,
                         context_window, **kwargs)
        self._noise_dist = self._get_noise_dist()
        self._noise_count = noise_count

    def _get_noise_dist(self):
        frequencies = list(self.summary.word_probability.values())
        frequencies.sort(reverse=True)
        smoothed_dist = np.power(frequencies, self.UNIGRAM_FACTOR)
        return smoothed_dist / np.sum(smoothed_dist)

    def get_batches(self) -> typing.Tuple[torch.LongTensor,
                                          torch.LongTensor,
                                          torch.LongTensor]:
        for idx in range(0, len(self._words), self._batch_size):
            target, context = [], []
            batch = self._words[idx:idx + self._batch_size]
            for i, word in enumerate(batch):
                word_token = self.summary.word_tokens[word]
                context_tokens = self._get_context_tokens(i,
                                                          self._context_window)
                target.extend([word_token] * len(context_tokens))
                context.extend(context_tokens)

            yield torch.LongTensor(target).to(self._device), torch.LongTensor(
                context).to(self._device), torch.LongTensor(
                np.random.choice(len(self._noise_dist),
                                 size=len(target) * self._noise_count,
                                 replace=True,
                                 p=self._noise_dist).reshape(
                    len(target), self._noise_count)).to(self._device)
