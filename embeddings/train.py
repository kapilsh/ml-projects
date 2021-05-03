from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from embeddings.loader import SGLoader, SGNSLoader
from embeddings.model import SkipGram, SkipGramNegativeSampling, \
    negative_sampling_loss, cosine_similarity


@dataclass
class HyperParameters:
    epochs: int
    embed_size: int
    lr: float
    snapshot_frequency: int


class SGModelTrainer:
    def __init__(self, save_path: str,
                 loader: SGLoader,
                 parameters: HyperParameters,
                 device: str = "cpu"):
        self._save_path = save_path
        self._loader = loader
        self._parameters = parameters
        self._device = device

    def train(self):
        steps = 0
        model = SkipGram(
            vocab_size=len(self._loader.summary.unique_words),
            embed_size=self._parameters.embed_size).to(self._device)

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._parameters.lr)

        best_loss = np.Inf
        loss_values = []

        for epoch in range(1, self._parameters.epochs + 1):
            logger.info(f"Started epoch {epoch}")
            for target, context in self._loader.get_batches():
                steps += 1
                outputs = model(target)
                loss = loss_function(outputs, context)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps % self._parameters.snapshot_frequency == 0:
                    if len(loss_values) == 0:
                        current_loss = loss.item()
                    else:
                        current_loss = loss_values[-1] + (
                                loss.item() - loss_values[-1]) / (
                                               len(loss_values) + 1)
                    logger.info(
                        f"[Epoch {epoch}] "
                        f"[Step {steps}]: Loss Value: {loss.item()}")
                    if loss_values and current_loss < loss_values[-1]:
                        logger.info(
                            f"Average Validation Loss Decreased "
                            f"{np.round(best_loss, 4)} => "
                            f"{np.round(current_loss, 4)} "
                            f"Saving model to {self._save_path}")
                        best_loss = current_loss
                        torch.save(model.state_dict(), self._save_path)

                    loss_values.append(current_loss)

    def test(self, count=10, top_k_count=5, random=True):
        model = SkipGram(
            vocab_size=len(self._loader.summary.unique_words),
            embed_size=self._parameters.embed_size)
        model.load_state_dict(torch.load(self._save_path))
        model = model.to(self._device)

        unique_words = self._loader.summary.unique_words
        token_count = len(unique_words)
        if random:
            tokens = np.random.choice(token_count, count, replace=True)
        else:
            tokens = np.arange(count)

        tokens = torch.LongTensor(tokens).to(self._device)

        return cosine_similarity(embedding=model.embed,
                                 tokens=tokens,
                                 unique_words=unique_words,
                                 top_k_count=top_k_count)


class SGNSModelTrainer:
    def __init__(self, save_path: str,
                 loader: SGNSLoader,
                 parameters: HyperParameters,
                 device: str = "cpu"):
        self._save_path = save_path
        self._loader = loader
        self._parameters = parameters
        self._device = device

    def train(self):
        steps = 0
        model = SkipGramNegativeSampling(
            vocab_size=len(self._loader.summary.unique_words),
            embed_size=self._parameters.embed_size).to(self._device)

        optimizer = optim.Adam(model.parameters(), lr=self._parameters.lr)
        best_loss = np.Inf
        loss_values = []

        for epoch in range(1, self._parameters.epochs + 1):
            logger.info(f"Started epoch {epoch}")
            for target, context, noise in self._loader.get_batches():
                steps += 1
                target_embed, context_embed, noise_embed = model(target,
                                                                 context,
                                                                 noise)
                loss = negative_sampling_loss(target=target_embed,
                                              context=context_embed,
                                              noise=noise_embed)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if steps % self._parameters.snapshot_frequency == 0:
                    current_loss = loss.item()
                    logger.info(
                        f"[Epoch {epoch}] "
                        f"[Step {steps}]: Loss Value: {loss.item()}")
                    if loss_values and current_loss < best_loss:
                        logger.info(
                            f"Average Validation Loss Decreased "
                            f"{np.round(best_loss, 4)} => "
                            f"{np.round(current_loss, 4)} "
                            f"Saving model to {self._save_path}")
                        best_loss = current_loss
                        torch.save(model.state_dict(), self._save_path)
                    loss_values.append(current_loss)

    def test(self, count=10, top_k_count=5, random=True):
        model = SkipGramNegativeSampling(
            vocab_size=len(self._loader.summary.unique_words),
            embed_size=self._parameters.embed_size)
        model.load_state_dict(torch.load(self._save_path))
        model = model.to(self._device)

        unique_words = self._loader.summary.unique_words
        token_count = len(unique_words)
        if random:
            tokens = np.random.choice(token_count, count, replace=True)
        else:
            tokens = np.arange(count)

        tokens = torch.LongTensor(tokens).to(self._device)
        return cosine_similarity(embedding=model.input_embedding,
                                 tokens=tokens,
                                 unique_words=unique_words,
                                 top_k_count=top_k_count)
