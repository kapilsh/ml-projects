import json
import os
from dataclasses import dataclass

import pandas as pd
import torch
from loguru import logger
from torch import nn
import torch.nn.functional as functional
import numpy as np

from rnn.loader import DataLoader, Tokens


class LSTMModel(nn.Module):
    def __init__(self, tokens: Tokens, **kwargs):
        super().__init__()
        self._drop_prob = kwargs.pop("drop_prob")
        self._hidden_size = kwargs.pop("hidden_size")
        self._num_layers = kwargs.pop("num_layers")
        self.tokens = tokens

        tokens_size = len(tokens.int_to_chars)

        self.lstm = nn.LSTM(
            input_size=tokens_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._drop_prob, batch_first=True)

        self.dropout = nn.Dropout(self._drop_prob)
        self.fc = nn.Linear(self._hidden_size, tokens_size)

    def forward(self, x, h, c):
        x_next, (hn, cn) = self.lstm(x, (h, c))
        x_dropout = self.dropout(x_next)
        x_stacked = x_dropout.contiguous().view(-1, self._hidden_size)
        output = self.fc(x_stacked)
        return output, hn, cn

    def initial_hidden_state(self, batch_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self._num_layers, batch_size,
                         self._hidden_size).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self._num_layers, batch_size,
                         self._hidden_size).requires_grad_()

        return h0, c0


@dataclass
class ModelHyperParameters:
    num_layers: int
    hidden_size: int
    epochs: int
    batch_size: int
    window: int
    learning_rate: float
    clip: float
    validation_split: float
    drop_prob: float
    validation_counts: int
    use_gpu: bool


def load_parameters(params_file):
    def read_json_file(file_path):
        with open(file_path) as json_file:
            return json.load(json_file)

    params_json = read_json_file(params_file)
    return ModelHyperParameters(**params_json)


class ModelRunner:
    def __init__(self, data_loader: DataLoader, save_path: str):
        self._data_loader = data_loader
        self._save_path = save_path

    def train(self, parameters: ModelHyperParameters):
        use_gpu = parameters.use_gpu and torch.cuda.is_available()
        if use_gpu:
            logger.info("GPU Available and Enabled: Using CUDA")
        else:
            logger.info("GPU Disabled: Using CPU")

        tokens = self._data_loader.tokens
        model = LSTMModel(tokens=tokens,
                          drop_prob=parameters.drop_prob,
                          num_layers=parameters.num_layers,
                          hidden_size=parameters.hidden_size)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parameters.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_data, valid_data = self._split_train_validation(
            tokens.tokens, parameters.validation_split)

        if use_gpu:
            model = model.cuda()

        n_chars = len(tokens.int_to_chars)

        losses = []

        for epoch in range(1, parameters.epochs + 1):
            runs = 0
            # initial hidden and cell state
            h, c = model.initial_hidden_state(parameters.batch_size)
            for x, y in DataLoader.generate_batches(train_data,
                                                    parameters.batch_size,
                                                    parameters.window):

                runs += 1

                x = DataLoader.one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y).view(
                    parameters.batch_size * parameters.window)

                if use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    h, c = h.cuda(), c.cuda()

                # detach - If we don't, we'll back-prop all the way to the start
                h, c = h.detach(), c.detach()

                model.zero_grad()
                output, h, c = model(inputs, h, c)

                loss = criterion(output, targets)
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)
                optimizer.step()

                if runs % parameters.validation_counts == 0:
                    # run validation
                    hv, cv = model.initial_hidden_state(parameters.batch_size)
                    validation_losses = []
                    model.eval()
                    for val_x, val_y in DataLoader.generate_batches(
                            valid_data, parameters.batch_size,
                            parameters.window):
                        inputs = torch.from_numpy(
                            DataLoader.one_hot_encode(val_x, n_chars))
                        targets = torch.from_numpy(val_y).view(
                            parameters.batch_size * parameters.window)

                        if use_gpu:
                            inputs, targets = inputs.cuda(), targets.cuda()
                            hv, cv = hv.cuda(), cv.cuda()

                        hv, cv = hv.detach(), cv.detach()

                        output, hv, cv = model(inputs, hv, cv)

                        val_loss = criterion(output, targets)
                        validation_losses.append(val_loss.item())

                    train_loss = loss.item()
                    val_loss_final = np.mean(validation_losses)

                    logger.info(
                        f"Epoch: {epoch}/{runs} | Training loss: {train_loss}"
                        f" | Validation loss: {val_loss_final}")

                    losses.append({
                        "Epoch": epoch,
                        "Run": runs,
                        "TrainLoss": train_loss,
                        "ValidationLoss": val_loss_final
                    })

                model.train()

            self._save_check_point(model, parameters, tokens, epoch)

        self._save_check_point(model, parameters, tokens)

        return pd.DataFrame(losses)

    def _save_check_point(self, model: LSTMModel,
                          parameters: ModelHyperParameters,
                          tokens: Tokens, epoch: int = None):
        epoch_str = str(epoch) if epoch else "final"
        file_path, file_ext = os.path.splitext(self._save_path)
        checkpoint_file = f"{file_path}_{epoch_str}{file_ext}"
        logger.info(f"Saving checkpoint to file {checkpoint_file}")
        result = {
            "parameters": parameters.__dict__,
            "model": model.state_dict(),
            "tokens": tokens
        }
        torch.save(result, checkpoint_file)

    @staticmethod
    def _split_train_validation(data: np.array, validation_split: float):
        total_count = len(data)
        train_count, validation_count = int(
            total_count * (1 - validation_split)), int(
            total_count * validation_split)
        return data[:train_count], data[train_count:]


def predict(model: LSTMModel, char: str, use_gpu: bool,
            h: torch.Tensor, c: torch.Tensor, top_k: int = 1):
    x = np.array([[model.tokens.chars_to_int[char]]])
    x = DataLoader.one_hot_encode(x, len(model.tokens.int_to_chars))
    inputs = torch.from_numpy(x)
    if use_gpu:
        inputs = inputs.cuda()
        model = model.cuda()

    h, c = h.detach(), c.detach()

    output, h, c = model(inputs, h, c)
    p = functional.softmax(output, dim=1).data

    if use_gpu:
        p = p.cpu()

    p, top_ch = p.topk(top_k)
    top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    char_token = np.random.choice(top_ch, p=p / p.sum())

    return model.tokens.int_to_chars[char_token], h, c


def generate_sample(model: LSTMModel, size: int, seed: str, top_k: int = 1,
                    use_gpu: bool = False):
    model.eval()  # eval mode

    text_chars = list(seed)
    h, c = model.initial_hidden_state(1)

    for i, char in enumerate(seed):
        next_char, h, c = predict(model=model, char=char,
                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)
        if i == len(seed) - 1:
            text_chars.append(next_char)

    for i in range(size):
        next_char, h, c = predict(model=model, char=text_chars[-1],
                                  use_gpu=use_gpu, h=h, c=c, top_k=top_k)
        text_chars.append(next_char)

    return ''.join(text_chars)
