# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

IMAGE_CLASSIFICATION_LABEL_NAME = "labels_mobilenet_quant_v1_224.txt"
IMAGE_CLASSIFICATION_MEDIA_NAME = "cat.jpg"
IMAGE_CLASSIFICATION_MODEL_NAME = "mobilenet_v1_1.0_224_quant.tflite"
IMAGE_CLASSIFICATION_MODEL_SHA1 = "765c995e1d27c4a738f77cf13445e7b41306befc"
IMAGE_CLASSIFICATION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                           "1yhQkJZwtuSyOvTOKi0ZVcIvZUR7GXv6z/" \
                                           "view?usp=sharing",
                                  'github': "https://github.com/diegohdorta/" \
                                            "models/raw/master/models/" \
                                            "eIQImageClassification.zip"}

FIRE_DETECTION_MEDIA_NAME = "fire.jpg"
FIRE_DETECTION_MODEL_NAME = "fire_detection.tflite"
FIRE_DETECTION_MODEL_SHA1 = "2df946680459a3b20bd668f423dcdaa6b76a98b3"
FIRE_DETECTION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                     "1aDsAB2i9wl1xbkexXbiYfPawKUMYXfwH/" \
                                     "view?usp=sharing",
                            'github': "https://github.com/diegohdorta/" \
                                      "models/raw/master/models/" \
                                      "eIQFireClassification.zip"}

CV_GREEN = (0, 255, 0)
CV_RED = (0, 0, 255)

FIRE = "Fire Detected!"
NO_FIRE = "No Fire"

ZIP = ".zip"

# Demos's Titles

TITLE_FIRE_CLASSIFICATION = "PyeIQ Fire Classification"
TITLE_IMAGE_CLASSIFICATION = "PyeIQ Image Classification"
