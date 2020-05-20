# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

OBJ_DETECTION_LABEL_NAME = "coco_labels.txt"
OBJ_DETECTION_MEDIA_NAME = "bus.jpg"
OBJ_DETECTION_MODEL_NAME = "detect.tflite"
OBJ_DETECTION_MODEL_SHA1 = "73b8bb0749f275c10366553bab6f5f313230c527"
OBJ_DETECTION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                   "1xdjdKPOH2PFPStbU2K2TEuFJN8Las3ag/" \
                                   "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "eIQObjectDetection.zip"}

OBJ_DETECTION_CV_GST_LABEL_NAME = "coco_labels.txt"
OBJ_DETECTION_CV_GST_MODEL_NAME = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
OBJ_DETECTION_CV_GST_MODEL_SHA1 = "ba623e959b743db24276fc91f5d9d081121f762f"
OBJ_DETECTION_CV_GST_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                          "1KF1hDfLvwJZ1S6102i8FvMwYoZmtEqLN/" \
                                          "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "mobilenet_ssd_v2_coco_quant.zip"}

OBJ_DETECTION_IMG_CAFFE_NAME = "MobileNetSSD_deploy.caffemodel"
OBJ_DETECTION_IMG_LABEL_NAME = "labels.txt"
OBJ_DETECTION_IMG_MEDIA_NAME = "dog.jpg"
OBJ_DETECTION_IMG_PROTO_NAME = "MobileNetSSD_deploy.prototxt"
OBJ_DETECTION_IMG_MODEL_SHA1 = "f9894307c83f8ddec91af76b8cd6f3dc07196dc0"
OBJ_DETECTION_IMG_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                       "1_qeq3CxK-xhrVX4qdsmnWQ_dwlMCWF76/" \
                                       "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "object_detection_image.zip"}

OBJ_DETECTION_SSD_LABEL_NAME = "coco_classes.txt"
OBJ_DETECTION_SSD_MEDIA_NAME = "dog.jpg"
OBJ_DETECTION_SSD_MODEL_NAME = "ssd_mobilenet_v2.tflite"
OBJ_DETECTION_SSD_MODEL_SHA1 = "fadfdb7c4bf056edee09cd37c87d06bb19e6ef83"
OBJ_DETECTION_SSD_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                       "1t3VmNdkpfp4M-jyz_AD2QUeyS78FyqQT/" \
                                       "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "object_detection_ssd.zip"}

ZIP = ".zip"

# Demos Titles

TITLE_OBJECT_DETECTION_CAM = "PyeIQ - Object Detection Camera"
TITLE_OBJECT_DETECTION_CV = "PyeIQ - Object Detection OpenCV"
TITLE_OBJECT_DETECTION_SSD = "PyeIQ - Object Detection SSD"
