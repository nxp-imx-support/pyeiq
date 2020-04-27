# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

OBJECT_RECOGNITION_MODEL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

LABEL_IMAGE_MODEL = "https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_quant.tflite"
LABEL_IMAGE_LABELS = "https://github.com/google-coral/edgetpu/raw/master/test_data/imagenet_labels.txt"
LABEL_IMAGE_DEFAULT_IMAGE = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp"

FIRE_DETECTION_MODEL = "1WDG7QEhP8zHPU1I8J55Ju7nZ7TxRjOeV"
FIRE_DETECTION_DEFAULT_IMAGE = "https://unsplash.com/photos/7WNEiyeMbB0/download?force=true&w=640"

CAMERA_OPENCV_MODEL = "https://dl.google.com/coral/canned_models/all_models.tar.gz"

CAMERA_OPENCV_DEFAULT_MODEL = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
CAMERA_OPENCV_DEFAULT_LABEL = "coco_labels.txt"

CAMERA_GSTREAMER_MODEL = "https://dl.google.com/coral/canned_models/all_models.tar.gz"

CAMERA_GSTREAMER_DEFAULT_MODEL = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
CAMERA_GSTREAMER_DEFAULT_LABEL = "coco_labels.txt"

# Apps Titles
TITLE_LABEL_IMAGE_SWITCH = "PyeIQ - Label Image Switch App"

# Demos Titles
TITLE_OBJECT_DETECTION_CAMERA = "PyeIQ - Object Detection Camera"
TITLE_OBJECT_DETECTION_OPENCV = "PyeIQ - Object Detection OpenCV"
TITLE_FIRE_DETECTION_CAMERA = "PyeIQ - Fire Detection Camera"
