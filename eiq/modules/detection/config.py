# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

LEFT = 0
TOP = 1
RIGHT = 2
BOTTOM = 3
CONFIDENCE = 4
CLASSES = 5

FACE_EYES_DETECTION = {'eye_cascade': "haarcascade_eye.xml",
                       'face_cascade': "haarcascade_frontalface_default.xml",
                       'image': "grace_hopper.jpg",
                       'sha1': "79ac7f90076ed5e2724e791bc27b3840ef63eb11",
                       'src': {'drive': "https://drive.google.com/file/d/"
                                        "1KS6nprGBr-JsuQmDQekvwhyGzBteXtfx/"
                                        "view?usp=sharing",
                               'github': "https://github.com/lizeze-0515/"
                                         "pymodel/releases/download/model_3.0.0/"
                                         "eIQFaceAndEyesDetection.zip"},
                       'window_title': "PyeIQ - Face and Eyes Detection"}

OBJ_DETECTION = {'image': "bus.jpg",
                 'labels': "labels_ssd_mobilenet_v1.txt",
                 'model': "ssd_mobilenet_v1_1_default_1.tflite",
                 'sha1': "73b8bb0749f275c10366553bab6f5f313230c527",
                 'src': {'drive': "https://drive.google.com/file/d/"
                                  "1LNo1JJwbLGpoTwgxEmKu8bjwrcdpdI4e/"
                                  "view?usp=sharing",
                         'github': "https://github.com/KaixinDing/"
                                   "pyeiq_model/releases/download/3.0.0/"
                                   "eIQObjectDetection.zip"},
                 'window_title': "PyeIQ - Object Detection"}

OBJ_DETECTION_GST = {'labels': "coco_labels.txt",
                     'model': "mobilenet_ssd_v2_coco_quant_postprocess.tflite",
                     'sha1': "4736e758d8d626047df7cd1b3c38c72e77fd32ee",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "14iRdCfznIlOYC4VURFH0acMHTJt0C4LH/"
                                      "view?usp=sharing",
                             'github': "https://github.com/lizeze-0515/"
                                       "pymodel/releases/download/model_3.0.0/"
                                       "eIQObjectDetectionCVGST.zip"},
                     'window_title': "PyeIQ - Object Detection OpenCV"}
