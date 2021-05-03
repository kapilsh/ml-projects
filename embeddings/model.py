import typing
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import numpy as np


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.data.uniform_(-1, 1)

        self.output = nn.Linear(embed_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.LongTensor) -> torch.LongTensor:
        x = self.embed(x)
        scores = self.output(x)
        log_likelihood = self.log_softmax(scores)
        # Alternatively, we can use the scores directly and
        # apply Cross-entropy loss
        return log_likelihood


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_size)
        self.output_embedding = nn.Embedding(vocab_size, embed_size)
        self.input_embedding.weight.data.uniform_(-1, 1)
        self.output_embedding.weight.data.uniform_(-1, 1)

    def forward(self, target: torch.LongTensor, context: torch.LongTensor,
                noise: torch.LongTensor) -> typing.Tuple[torch.LongTensor,
                                                         torch.LongTensor,
                                                         torch.LongTensor]:
        target_embed = self.input_embedding(target)
        context_embed = self.output_embedding(context)
        noise_embed = self.output_embedding(noise)
        return target_embed, context_embed, noise_embed


def negative_sampling_loss(target: torch.LongTensor,
                           context: torch.LongTensor,
                           noise: torch.LongTensor):
    batch_size, embed_size = target.shape
    target = target.view(batch_size, embed_size, 1)
    context = context.view(batch_size, 1, embed_size)

    output_loss = torch.bmm(context, target).sigmoid().log().squeeze()
    noise_loss = torch.bmm(noise.neg(),
                           target).sigmoid().log().squeeze().sum(1)
    return -(output_loss + noise_loss).mean()


@dataclass
class CosineSimilarity:
    words: typing.List[str]
    similarities: torch.Tensor
    similar_words: typing.List[List[str]]
    similar_words_scores: torch.Tensor


def cosine_similarity(embedding: nn.Embedding,
                      tokens: torch.LongTensor,
                      unique_words: List[str],
                      top_k_count: int) -> CosineSimilarity:
    embedding_weights = embedding.weight
    embedding_norms = embedding_weights.pow(2).sum(dim=1).sqrt()
    normed_embedding = (embedding_weights.t() / embedding_norms).t()

    example_words = [unique_words[token] for token in tokens]
    example_embeddings = embedding(tokens)
    similarities = torch.mm(example_embeddings, normed_embedding.t())

    scores, top_k_similar = similarities.topk(top_k_count)
    similar_words = np.array(unique_words)[top_k_similar.cpu().numpy()][:, 1:]

    return CosineSimilarity(words=example_words,
                            similarities=similarities,
                            similar_words=similar_words,
                            similar_words_scores=scores)
