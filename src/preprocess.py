import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(data_path, limit=100):
    classes = ["yes", "no"]

    images = []
    labels = []

    for label, category in enumerate(classes):
        folder = os.path.join(data_path, category)

        count = 0

        for image_name in os.listdir(folder):

            if count >= limit:
                break

            image_path = os.path.join(folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = image / 255.0

            images.append(image)
            labels.append(label)

            count += 1

    return np.array(images), np.array(labels)