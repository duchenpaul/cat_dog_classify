import cv2
import numpy as np
from matplotlib import pyplot as plt

import config


IMG_SIZE = config.IMG_SIZE


def img_read(img_path):
    return cv2.imread(img_path)


def img_normalize(img):
    # TODO: No need to cvtColor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img


def image_process(img_path):
    return img_normalize(img_read(img_path))


if __name__ == '__main__':
    import os
    img_path = os.path.join(config.DATASET_DIR, 'image_1051.jpg')
    img = cv2.imread(img_path)
    img = img_normalize(img)
    print(type(img))
    plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
    plt.show()