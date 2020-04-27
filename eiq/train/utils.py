# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import config
import numpy as np
import cv2
from imutils import paths
import os
import pathlib
import logging
logging.basicConfig(level=logging.INFO)


def log(*args):
    logging.info(" ".join("%s" % a for a in args))


def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)


def load_dataset(dataset_path):
    image_paths = list(paths.list_images(dataset_path))
    data = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        data.append(image)
    return np.array(data, dtype="float32")
