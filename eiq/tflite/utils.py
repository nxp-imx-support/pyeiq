import os
import sys
import shutil

from pathlib import Path

def get_model(model_path: str = None, model: str = None):

    path = os.path.dirname(model_path)
    
    shutil.unpack_archive(model_path, path)

    for p in Path(path).rglob('*.tflite'):
        model = str(p)

    return model

def get_label(label_path: str = None, label: str = None):

    path = os.path.dirname(label_path)
    
    shutil.unpack_archive(label_path, path)

    for p in Path(path).rglob('*.txt'):
        if "label" in str(p):
            label = str(p)

    return label
