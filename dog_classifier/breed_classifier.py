import os
from collections import namedtuple

import click
from PIL.ImageFile import ImageFile
from loguru import logger
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as functional
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DataProvider:
    def __init__(self, root_dir: str, **kwargs):
        self._root_dir = root_dir
        self._train_subfolder = kwargs.pop("train_subfolder", "train")
        self._test_subfolder = kwargs.pop("test_subfolder", "test")
        self._validation_subfolder = kwargs.pop("validation_subfolder", "valid")

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        transform_others = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self._train_loader = DataLoader(
            datasets.ImageFolder(os.path.join(root_dir, self._train_subfolder),
                                 transform=transform_train), **kwargs)

        self._validation_loader = DataLoader(
            datasets.ImageFolder(
                os.path.join(root_dir, self._validation_subfolder),
                transform=transform_others), **kwargs)

        self._test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(root_dir, self._test_subfolder),
                                 transform=transform_others), **kwargs)

    @property
    def train(self) -> DataLoader:
        return self._train_loader

    @property
    def test(self) -> DataLoader:
        return self._test_loader

    @property
    def validation(self) -> DataLoader:
        return self._validation_loader


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Conv Layers
        self.conv1 = nn.Conv2d(3, 16, (3,))
        self.conv2 = nn.Conv2d(16, 32, (3,))
        self.conv3 = nn.Conv2d(32, 64, (3,))
        self.conv4 = nn.Conv2d(64, 128, (3,))
        self.conv5 = nn.Conv2d(128, 256, (3,))

        # Pooling layer
        self.max_pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(5 * 5 * 256, 400)
        self.fc2 = nn.Linear(400, 133)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.max_pool(functional.relu(self.conv1(x)))
        x = self.max_pool(functional.relu(self.conv2(x)))
        x = self.max_pool(functional.relu(self.conv3(x)))
        x = self.max_pool(functional.relu(self.conv4(x)))
        x = self.max_pool(functional.relu(self.conv5(x)))

        x = x.view(-1, 5 * 5 * 256)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


TrainedModel = namedtuple(
    "TrainedModel", ["model", "validation_losses", "optimal_validation_loss"])

TestResult = namedtuple(
    "TestResult", ["test_loss", "correct_labels", "total_labels"])


class Model:
    def __init__(self, data_provider: DataProvider,
                 save_path: str, **kwargs):
        self._data_provider = data_provider
        self._save_path = save_path
        self._criterion = nn.CrossEntropyLoss()
        self._use_gpu = kwargs.pop("use_gpu",
                                   False) and torch.cuda.is_available()
        if self._use_gpu:
            logger.info("CUDA is enabled - using GPU")

    def train(self, n_epochs) -> TrainedModel:
        neural_net = NeuralNet()
        optimizer = optim.Adam(neural_net.parameters())
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        validation_losses = []
        min_validation_loss = np.Inf

        for epoch in range(n_epochs):
            validation_loss = self._train_epoch(epoch, neural_net, optimizer)
            validation_losses.append(validation_loss)
            if min_validation_loss > validation_loss:
                min_validation_loss = validation_loss
                logger.info(
                    "Validation Loss Decreased: {:.6f} => {:.6f}. "
                    "Saving Model to {}".format(
                        min_validation_loss, validation_loss, self._save_path))
                torch.save(neural_net.state_dict(), self._save_path)

        optimal_model = NeuralNet()
        optimal_model.load_state_dict(torch.load(self._save_path))
        return TrainedModel(model=optimal_model,
                            validation_losses=validation_losses,
                            optimal_validation_loss=min_validation_loss)

    def _train_epoch(self, epoch: int, neural_net: nn.Module,
                     optimizer: optim.Optimizer):
        train_loss = 0
        logger.info(f"[Epoch {epoch}] Starting training phase")
        neural_net.train()
        for batch_index, data, target in enumerate(
                self._data_provider.train):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = neural_net(data)
            loss = self._criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + (
                    (loss.item() - train_loss) / (batch_index + 1))

        logger.info(f"[Epoch {epoch}] Starting eval phase")

        validation_loss = 0
        neural_net.eval()
        for batch_index, data, target in enumerate(
                self._data_provider.validation):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = neural_net(data)
            loss = self._criterion(output, target)
            validation_loss = validation_loss + (
                    (loss.item() - validation_loss) / (batch_index + 1))

        return validation_loss

    def test(self) -> TestResult:
        model = NeuralNet()
        model.load_state_dict(torch.load(self._save_path))
        test_loss = 0
        correct_labels = 0
        total_labels = 0
        model.eval()
        for batch_idx, data, target in enumerate(self._data_provider.test):
            # move to GPU
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = self._criterion(output, target)
            test_loss = test_loss + ((loss.data - test_loss) / (batch_idx + 1))
            pred = output.data.max(1, keepdim=True)[1]
            correct_labels += np.sum(
                np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total_labels += data.size(0)

        return TestResult(test_loss=test_loss, correct_labels=correct_labels,
                          total_labels=total_labels)


@click.group()
def train_cli():
    pass


@train_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--epochs", required=True, type=int)
@click.option("--gpu/--no-gpu", default=False)
def train(data_path: str, save_path: str, epochs: int, gpu: bool):
    data_provider = DataProvider(root_dir=data_path)
    model = Model(data_provider=data_provider, use_gpu=gpu,
                  save_path=save_path)
    model.train(n_epochs=epochs)


@click.group()
def test_cli():
    pass


@test_cli.command()
@click.option("--data-path", required=True, type=str)
@click.option("--save-path", required=True, type=str)
@click.option("--gpu/--no-gpu", default=False)
def test(data_path: str, save_path: str, gpu: bool):
    data_provider = DataProvider(root_dir=data_path)
    model = Model(data_provider=data_provider, use_gpu=gpu,
                  save_path=save_path)
    model.test()


main = click.CommandCollection(sources=[train_cli, test_cli])

if __name__ == '__main__':
    main()
