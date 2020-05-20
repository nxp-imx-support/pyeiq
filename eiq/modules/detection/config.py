# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

OBJ_DETECTION_LABEL_NAME = "coco_labels.txt"
OBJ_DETECTION_MEDIA_NAME = "bus.jpg"
OBJ_DETECTION_MODEL_NAME = "detect.tflite"
OBJ_DETECTION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                   "1xdjdKPOH2PFPStbU2K2TEuFJN8Las3ag/" \
                                   "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "eIQObjectDetection.zip"}

OBJ_DETECTION_CV_GST_LABEL_NAME = "coco_labels.txt"
OBJ_DETECTION_CV_GST_MODEL_NAME = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
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
OBJ_DETECTION_IMG_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                       "1_qeq3CxK-xhrVX4qdsmnWQ_dwlMCWF76/" \
                                       "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "object_detection_image.zip"}

OBJ_DETECTION_SSD_LABEL_NAME = "coco_classes.txt"
OBJ_DETECTION_SSD_MEDIA_NAME = "dog.jpg"
OBJ_DETECTION_SSD_MODEL_NAME = "ssd_mobilenet_v2.tflite"
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
