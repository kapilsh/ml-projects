import abc
import os
from abc import abstractmethod
from collections import namedtuple
from typing import Tuple

from loguru import logger
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as functional
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import ImageFile
from tqdm import tqdm

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

        normalization_means = kwargs.pop("norm_means")
        normalization_stds = kwargs.pop("norm_stds")

        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(normalization_means, normalization_stds)
        ])

        transform_others = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(normalization_means, normalization_stds)
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
    "TestResult", ["test_loss", "correct_labels", "total_labels",
                   "predicted_labels", "target_labels"])


class BaseModel(metaclass=abc.ABCMeta):
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

        self._verbose = kwargs.pop("verbose", False)

    @property
    @abstractmethod
    def train_model(self) -> nn.Module:
        raise NotImplementedError("Implement in derived class")

    @property
    @abstractmethod
    def test_model(self) -> nn.Module:
        raise NotImplementedError("Implement in base class")

    def train(self, n_epochs: int) -> TrainedModel:
        model = self.train_model
        optimizer = optim.Adam(model.parameters())
        logger.info(f"Model Architecture: \n{model}")

        validation_losses = []
        train_losses = []
        min_validation_loss = np.Inf

        for epoch in range(n_epochs):
            train_loss, validation_loss = self._train_epoch(epoch, model,
                                                            optimizer)
            validation_losses.append(validation_loss)
            train_losses.append(train_loss)
            if min_validation_loss > validation_loss:
                logger.info(
                    "Validation Loss Decreased: {:.6f} => {:.6f}. "
                    "Saving Model to {}".format(
                        min_validation_loss, validation_loss, self._save_path))
                min_validation_loss = validation_loss
                torch.save(model.state_dict(), self._save_path)

        return TrainedModel(train_losses=train_losses,
                            validation_losses=validation_losses,
                            optimal_validation_loss=min_validation_loss)

    def _train_epoch(self, epoch: int, neural_net: nn.Module,
                     optimizer: optim.Optimizer) -> Tuple[float, float]:
        train_loss = 0
        logger.info(f"[Epoch {epoch}] Starting training phase")
        neural_net.train()
        total_samples = len(self._data_provider.train.dataset.samples)
        batch_count = (total_samples // self._data_provider.train.batch_size)
        for batch_index, (data, target) in tqdm(enumerate(
                self._data_provider.train), total=batch_count + 1, ncols=80):
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
        total_samples = len(self._data_provider.validation.dataset.samples)
        batch_count = (
                total_samples // self._data_provider.validation.batch_size)
        neural_net.eval()
        for batch_index, (data, target) in tqdm(enumerate(
                self._data_provider.validation), total=batch_count + 1,
                ncols=80):
            if self._use_gpu:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = neural_net(data)
            loss = self._criterion(output, target)
            validation_loss = validation_loss + (
                    (loss.item() - validation_loss) / (batch_index + 1))

        return train_loss, validation_loss

    def test(self) -> TestResult:
        model = self.test_model
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
            predicted_labels = np.append(predicted_labels,
                                         predicted.cpu().numpy())
            target_labels = np.append(target_labels, target.cpu().numpy())

        return TestResult(test_loss=test_loss,
                          correct_labels=sum(np.equal(target_labels,
                                                      predicted_labels)),
                          total_labels=len(target_labels),
                          predicted_labels=predicted_labels,
                          target_labels=target_labels)


class ModelScratch(BaseModel):

    @property
    def train_model(self) -> nn.Module:
        neural_net = NeuralNet()
        if self._use_gpu:
            neural_net = neural_net.cuda()
        return neural_net

    @property
    def test_model(self) -> nn.Module:
        model = NeuralNet()
        model.load_state_dict(torch.load(self._save_path))
        if self._use_gpu:
            model = model.cuda()
        return model


class ModelTransferLearn(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transfer_model = self._get_model_arch()

    def _get_model_arch(self) -> nn.Module:
        vgg16 = models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.requires_grad = False  # pre-trained - dont touch
        n_inputs_final_layer = vgg16.classifier[-1].in_features
        n_classes = len(self._data_provider.train.dataset.classes)
        # Replace the final layer
        final_layer = nn.Linear(n_inputs_final_layer, n_classes)
        vgg16.classifier[-1] = final_layer
        return vgg16

    @property
    def train_model(self) -> nn.Module:
        if self._use_gpu:
            return self._transfer_model.cuda()
        return self._transfer_model

    @property
    def test_model(self) -> nn.Module:
        test_model_arch = self._get_model_arch()
        test_model_arch.load_state_dict(torch.load(self._save_path))
        if self._use_gpu:
            return test_model_arch.cuda()
        return test_model_arch
