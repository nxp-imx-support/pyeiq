# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

IMAGE_CLASSIFICATION_LABEL_NAME = "labels_mobilenet_quant_v1_224.txt"
IMAGE_CLASSIFICATION_MEDIA_NAME = "cat.jpg"
IMAGE_CLASSIFICATION_MODEL_NAME = "mobilenet_v1_1.0_224_quant.tflite"
IMAGE_CLASSIFICATION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                           "1yhQkJZwtuSyOvTOKi0ZVcIvZUR7GXv6z/" \
                                           "view?usp=sharing",
                                  'github': "https://github.com/diegohdorta/" \
                                            "models/raw/master/models/" \
                                            "eIQImageClassification.zip"}

LABEL_IMAGE_LABEL_NAME = "imagenet_labels.txt"
LABEL_IMAGE_MEDIA_NAME = "grace_hopper.bmp"
LABEL_IMAGE_MODEL_NAME = "mobilenet_v2_1.0_224_quant.tflite"
LABEL_IMAGE_MODEL_SRC = {'drive':  "https://drive.google.com/file/d/" \
                                   "1mKCQ6ji5ZQ1IMkZ8HHV3q5CGl1OiNgz-/" \
                                   "view?usp=sharing",
                         'github': "https://github.com/diegohdorta/" \
                                   "models/raw/master/models/" \
                                   "label_image.zip"}

FIRE_DETECTION_MEDIA_NAME = "fire.jpg"
FIRE_DETECTION_MODEL_NAME = "fire_detection.tflite"
FIRE_DETECTION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                     "1aDsAB2i9wl1xbkexXbiYfPawKUMYXfwH/" \
                                     "view?usp=sharing",
                            'github': "https://github.com/diegohdorta/" \
                                      "models/raw/master/models/" \
                                      "eIQFireClassification.zip"}

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CV_GREEN = (0, 255, 0)
CV_RED = (0, 0, 255)

FIRE = "Fire Detected!"
NO_FIRE = "No Fire"

ZIP = ".zip"

# Demos's Titles

TITLE_FIRE_CLASSIFICATION = "PyeIQ Fire Classification"
TITLE_IMAGE_CLASSIFICATION = "PyeIQ Image Classification"
