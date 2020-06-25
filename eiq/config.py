# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os

import cv2


INIT_MODULE_FILE = "__init__.py"

TMP_FILE_PATH = "eiq"
HOME = os.environ['HOME']
BASE_DIR = os.path.join(HOME, ".cache", "eiq")

CHUNK_DEFAULT_SIZE = 32768
REGULAR_DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

INF_TIME_MSG = "INFERENCE TIME"
FPS_MSG = "FPS"
MODEL_MSG = "MODEL"
SRC_MSG = "SOURCE"
MAX_TIME = datetime.timedelta(9, 9, 9)
ID = 5

ZIP = ".zip"

FONT = {'hershey': cv2.FONT_HERSHEY_SIMPLEX,
        'size': 0.8,
        'color': {'black': (0, 0, 0),
                  'blue': (255, 0, 0),
                  'green': (0, 255, 0),
                  'orange': (0, 127, 255),
                  'red': (0, 0, 255),
                  'white': (255, 255, 255)},
        'thickness': 2}
