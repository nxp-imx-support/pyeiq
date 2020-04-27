# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import shutil

from pathlib import Path


def get_model_from_zip(model_path: str = None):
    model = None

    path = os.path.dirname(model_path)
    shutil.unpack_archive(model_path, path)

    for p in Path(path).rglob('*.tflite'):
        model = str(p)

    return model

def get_model_from_path(model_path: str = None):
    model = None

    for p in Path(model_path).rglob('*.tflite'):
        model = str(p)

    return model



def get_label(label_path: str = None):
    label = None

    path = os.path.dirname(label_path)
    shutil.unpack_archive(label_path, path)

    for p in Path(path).rglob('*.txt'):
        if "label" in str(p):
            label = str(p)

    return label
