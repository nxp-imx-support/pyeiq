# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

TMP_FILE_PATH = "train"

GD_ID_FIRE_DATASET = "1e_8VgBzBjjBftlIWtSmoL2oYPu0Vzrs8"
GD_ID_NON_FIRE_DATASET = "1xk5R0Ulo1nnPgoVZvI82ag3YiKRdTRRN"

TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

CLASSES = ["Non-Fire", "Fire"]

MODEL_PATH_PB_FORMAT = os.path.sep.join(["output", "fire_detection.model"])
MODEL_PATH_H5_FORMAT = os.path.sep.join(
    ["output", "fire_detection.model", "saved_model.h5"])
