import os
from collections import namedtuple

from loguru import logger
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as functional
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataProvider:
    def __init__(self, root_dir: str, **kwargs):
        self._root_dir = root_dir
        self._train_subfolder = kwargs.pop("train_subfolder", "train")
        self._test_subfolder = kwargs.pop("test_subfolder", "test")
        self._validation_subfolder = kwargs.pop("validation_subfolder", "valid")
        self._batch_size = kwargs.pop("batch_size", 64)
        self._num_workers = kwargs.pop("num_workers", 0)

        logger.info(f"ROOT_DIR: {self._root_dir}")
        logger.info(f"BATCH_SIZE: {self._batch_size}")
        logger.info(f"NUM WORKERS: {self._num_workers}")

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
            datasets.ImageFolder(
                os.path.join(root_dir, self._train_subfolder),
                transform=transform_train),
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

        self._validation_loader = DataLoader(
            datasets.ImageFolder(
                os.path.join(root_dir, self._validation_subfolder),
                transform=transform_others),
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

        self._test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(root_dir, self._test_subfolder),
                                 transform=transform_others),
            shuffle=False,
            batch_size=self._batch_size,
            num_workers=self._num_workers)

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

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, (3, 3))
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5 * 5 * 256, 400)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(400, 133)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(functional.relu(x))
        x = self.conv2(x)
        x = self.pool2(functional.relu(x))
        x = self.conv3(x)
        x = self.pool3(functional.relu(x))
        x = self.conv4(x)
        x = self.pool4(functional.relu(x))
        x = self.conv5(x)
        x = self.pool5(functional.relu(x))

        x = x.view(-1, 5 * 5 * 256)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


TrainedModel = namedtuple(
    "TrainedModel",
    ["train_losses", "validation_losses", "optimal_validation_loss"])

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
        else:
            logger.info("GPU Disabled: Using CPU")

    def train(self, n_epochs) -> TrainedModel:

        neural_net = NeuralNet()
        if self._use_gpu:
            neural_net = neural_net.cuda()
        logger.info(f"Model Architecture: \n{neural_net}")
        optimizer = optim.Adam(neural_net.parameters())
        validation_losses = []
        train_losses = []
        min_validation_loss = np.Inf

        for epoch in range(n_epochs):
            train_loss, validation_loss = self._train_epoch(epoch, neural_net,
                                                            optimizer)
            validation_losses.append(validation_loss)
            train_losses.append(train_loss)
            if min_validation_loss > validation_loss:
                logger.info(
                    "Validation Loss Decreased: {:.6f} => {:.6f}. "
                    "Saving Model to {}".format(
                        min_validation_loss, validation_loss, self._save_path))
                min_validation_loss = validation_loss
                torch.save(neural_net.state_dict(), self._save_path)

        optimal_model = NeuralNet()
        optimal_model.load_state_dict(torch.load(self._save_path))
        return TrainedModel(train_losses=train_losses,
                            validation_losses=validation_losses,
                            optimal_validation_loss=min_validation_loss)

    def _train_epoch(self, epoch: int, neural_net: nn.Module,
                     optimizer: optim.Optimizer):
        train_loss = 0
        logger.info(f"[Epoch {epoch}] Starting training phase")
        neural_net.train()
        for batch_index, (data, target) in enumerate(
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
        for batch_index, (data, target) in enumerate(
                self._data_provider.validation):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = neural_net(data)
            loss = self._criterion(output, target)
            validation_loss = validation_loss + (
                    (loss.item() - validation_loss) / (batch_index + 1))

        return train_loss, validation_loss

    def test(self) -> TestResult:
        model = NeuralNet()
        model.load_state_dict(torch.load(self._save_path))
        if self._use_gpu:
            model = model.cuda()
        test_loss = 0
        predicted_labels = np.array([])
        target_labels = np.array([])

        model.eval()
        for batch_idx, (data, target) in enumerate(self._data_provider.test):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = self._criterion(output, target)
            test_loss = test_loss + (
                    (loss.data.item() - test_loss) / (batch_idx + 1))
            predicted = output.max(1).indices
            predicted_labels = np.append(predicted_labels, predicted.numpy())
            target_labels = np.append(target_labels, target.numpy())

        return TestResult(test_loss=test_loss,
                          correct_labels=sum(np.equal(target_labels,
                                                      predicted_labels)),
                          total_labels=len(target_labels))
