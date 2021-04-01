from glob import glob
import time

import cv2
from loguru import logger
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# %%


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f"{method.__qualname__} Took {int((te - ts) * 1000)}ms")
        return result

    return timed

#%%


class DogDetector:
    IMAGENET_MIN_INDEX_DOG = 151
    IMAGENET_MAX_INDEX_DOG = 268

    def __init__(self, use_gpu: bool = False):
        self._model = models.vgg16(pretrained=True)
        self._use_cuda = torch.cuda.is_available() and use_gpu
        if self._use_cuda:
            logger.info("CUDA is enabled - using GPU")
            self._model = self._model.cuda()

    @staticmethod
    def _read_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # def predict(self, image_path: str) -> int:
    #     image = Image.open(image_path)
    #
    #     preprocess = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    #     ])
    #
    #     image = preprocess(image).unsqueeze_(0)
    #
    #     if self._use_cuda:
    #         image = image.cuda()
    #
    #     with torch.no_grad():
    #         output = self._model(image)
    #         predicted = output.argmax()
    #
    #     return predicted

    def predict(self, image_path: str) -> int:
        image = self._read_image(image_path)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = preprocess(image).unsqueeze_(0)

        if self._use_cuda:
            image = image.cuda()

        with torch.no_grad():
            output = self._model(image)
            predicted = output.argmax()

        return predicted

    def detect(self, image_path: str) -> bool:
        predicted_index = self.predict(image_path)
        logger.info(f"Predicted Index: {predicted_index}")
        return (self.IMAGENET_MIN_INDEX_DOG <= predicted_index <=
                self.IMAGENET_MAX_INDEX_DOG)


# %%


if __name__ == '__main__':
    dog_files = np.array(glob("/data/dog_images/*/*/*"))
    dog_detector = DogDetector(use_gpu=False)
    chosen_size = 100
    detected = sum(dog_detector.detect(f) for f in dog_files[:chosen_size])
    logger.info(f"Dogs detected in {detected} / {chosen_size} = "
                f"{detected * 100 / chosen_size}% images")

    human_files = np.array(glob("/data/lfw/*/*"))
    detected = sum(dog_detector.detect(f) for f in human_files[:chosen_size])
    logger.info(f"Dogs detected in {detected} / {chosen_size} = "
                f"{detected * 100 / chosen_size}% images")