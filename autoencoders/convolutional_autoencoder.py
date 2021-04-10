from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class DataProvider:
    def __init__(self, root_dir, batch_size, num_workers=0):
        self._root_dir = root_dir
        transform = transforms.ToTensor()
        train_data = datasets.FashionMNIST(root=root_dir, train=True,
                                           download=True,
                                           transform=transform)
        test_data = datasets.FashionMNIST(root=root_dir, train=False,
                                          download=True,
                                          transform=transform)
        self._train_loader = DataLoader(train_data, batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True)
        self._test_loader = DataLoader(test_data, batch_size=batch_size,
                                       num_workers=num_workers, shuffle=True)

    @property
    def train(self) -> DataLoader:
        return self._train_loader

    @property
    def test(self) -> DataLoader:
        return self._test_loader


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 4, (3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(4, 16, (2, 2), stride=(2, 2))
        self.t_conv2 = nn.ConvTranspose2d(16, 1, (2, 2), stride=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_conv1(x)
        x = functional.relu(x)
        x = self.t_conv2(x)
        x = functional.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


TrainedModel = namedtuple("TrainedModel", ["model", "train_losses"])


class Model:
    def __init__(self, data_provider: DataProvider, use_gpu: bool = False):
        self._data_provider = data_provider
        self._criterion = nn.MSELoss()
        self._use_gpu = use_gpu and torch.cuda.is_available()

    def train(self, n_epochs: int) -> TrainedModel:
        model = AutoEncoder()
        if self._use_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        for epoch in range(1, n_epochs + 1):
            logger.info(f"[EPOCH {epoch}: Starting training")
            train_loss = 0.0
            for data, _ in self._data_provider.train:
                optimizer.zero_grad()
                if self._use_gpu:
                    data = data.cuda()
                output = model(data)
                loss = self._criterion(output, data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            train_loss = train_loss / len(self._data_provider.train)
            logger.info(
                f"[EPOCH {epoch}: Training loss {np.round(train_loss, 6)}")
            train_losses.append(train_loss)

        return TrainedModel(model=model, train_losses=train_losses)

    def test(self, model: AutoEncoder):
        if self._use_gpu:
            model = model.cuda()
        model.eval()
        for data, _ in self._data_provider.test:
            result = model(data)
            yield data, result
