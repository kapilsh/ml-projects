from collections import namedtuple
from typing import List

from loguru import logger

import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt

# %%

DetectedFace = namedtuple("DetectedFace", ["faces", "image"])


def detect_faces(file_name: str) -> DetectedFace:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    logger.info(f'Faces detected: {faces}')

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return DetectedFace(faces=faces, image=cv_rgb)


def plot_detected_faces(img: DetectedFace):
    fig, ax = plt.subplots()
    ax.imshow(img.image)
    plt.show()


def face_present(file_path: str) -> bool:
    img = detect_faces(file_path)
    return len(img.faces) > 0


def plot_detected_faces_multiple(results: List[DetectedFace],
                                 rows: int = 3, columns: int = 3):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    for r in range(rows):
        for c in range(columns):
            img = results[r * columns + c]
            ax[r][c].imshow(img.image)
    plt.show()


# %%

if __name__ == '__main__':
    human_files = np.array(glob("/data/lfw/*/*"))
    dog_files = np.array(glob("/data/dog_images/*/*/*"))

    logger.info("Detect random human image")
    image = detect_faces(human_files[np.random.randint(0, len(human_files))])
    plot_detected_faces(image)

    logger.info("Detect random dog image")
    image = detect_faces(dog_files[np.random.randint(0, len(dog_files))])
    plot_detected_faces(image)

    filter_count = 1000

    human_images_result = list(map(
        detect_faces,
        human_files[np.random.randint(0, len(human_files), filter_count)]))
    dog_images_result = list(map(
        detect_faces,
        dog_files[np.random.randint(0, len(dog_files), filter_count)]))

    plot_detected_faces_multiple(human_images_result)
    plot_detected_faces_multiple(dog_images_result)
