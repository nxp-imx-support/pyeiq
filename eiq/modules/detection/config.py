# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import cv2


EMOTIONS_DETECTION_CASCADE_FACE_NAME = "haarcascade_frontalface_default.xml"
EMOTIONS_DETECTION_MEDIA_NAME = "grace_hopper.jpg"
EMOTIONS_DETECTION_MODEL_NAME = "model.tflite"
EMOTIONS_DETECTION_SHA1 = "d112d8e378f8457a48b66aea80bc4e714e8e2e41"
EMOTIONS_DETECTION_SRC = {'drive': "https://drive.google.com/file/d/" \
                                    "15VTXVJ7GwS_Rr2vvl9xve9UAcFO3LFzG/" \
                                    "view?usp=sharing",
                           'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "eIQEmotionsDetection.zip"}

FACE_EYES_DETECTION_CASCADE_EYES_NAME = "haarcascade_eye.xml"
FACE_EYES_DETECTION_CASCADE_FACE_NAME = "haarcascade_frontalface_default.xml"
FACE_EYES_DETECTION_MEDIA_NAME = "grace_hopper.jpg"
FACE_EYES_DETECTION_SHA1 = "79ac7f90076ed5e2724e791bc27b3840ef63eb11"
FACE_EYES_DETECTION_SRC = {'drive': "https://drive.google.com/file/d/" \
                                    "1HPWV4W4FnrfG14tnZF7bugkgoCmMpM-p/" \
                                    "view?usp=sharing",
                           'github': "https://github.com/diegohdorta/" \
                                        "models/raw/master/models/" \
                                        "eIQFaceAndEyesDetection.zip"}

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
                                        
YOLOV3_OBJ_DETECTION_LABEL_NAME = "labels.txt"
YOLOV3_OBJ_DETECTION_MEDIA_NAME = "example.jpg"
YOLOV3_OBJ_DETECTION_MODEL_NAME = "tiny_yolov3.tflite"
YOLOV3_OBJ_DETECTION_MODEL_SHA1 = "406438b9a5a530f6f6874341219a749e4f209b6e"
YOLOV3_OBJ_DETECTION_MODEL_SRC = {'drive': "https://drive.google.com/file/d/" \
                                           "1gzUVbyDZrgAFDyZG3aRpnsClufL9Q7Hx/" \
                                           "view?usp=sharing",
                                  'github': "https://github.com/diegohdorta/" \
                                            "models/raw/master/models/"\
                                            "eIQObjectDetectionYOLOV3.zip"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.8
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 2

ZIP = ".zip"

# Demos Titles

TITLE_EMOTIONS_DETECTION = "PyeIQ - Emotions Detection"
TITLE_FACE_EYES_DETECTION = "PyeIQ - Face and Eyes Detection"
TITLE_OBJECT_DETECTION_CAM = "PyeIQ - Object Detection Camera"
TITLE_OBJECT_DETECTION_CV = "PyeIQ - Object Detection OpenCV"
TITLE_OBJECT_DETECTION_SSD = "PyeIQ - Object Detection SSD"
TITLE_OBJECT_DETECTION_YOLOV3 = "PyeIQ - Object Detection YOLOV3"
