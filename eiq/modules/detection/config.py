# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

LEFT = 0
TOP = 1
RIGHT = 2
BOTTOM = 3
CONFIDENCE = 4
CLASSES = 5

OBJ_DETECTION = {'image': "bus.jpg",
                 'labels': "labels_ssd_mobilenet_v1.txt",
                 'model': "ssd_mobilenet_v1_1_default_1.tflite",
                 'sha1': "73b8bb0749f275c10366553bab6f5f313230c527",
                 'src': {'drive': "https://drive.google.com/file/d/"
                                  "1rdq36z1qpCz1GZUslqEVtenTReMF51TH/"
                                  "view?usp=sharing",
                         'github': "https://github.com/KaixinDing/"
                                   "pyeiq_model/releases/download/3.0.0/"
                                   "eIQObjectDetection.zip"},
                 'window_title': "PyeIQ - Object Detection"}

