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
                 'sha1': "eef8713a6507850998e9e405f0a2b02a79fba757",
                 'src': {'drive': "https://drive.google.com/file/d/"
                                  "1qwpKsDiKYttEqvoOPWgY8d8FSIyFgNGR",
                         'github': "https://github.com/KaixinDing/"
                                   "pyeiq_model/releases/download/3.0.0/"
                                   "eIQObjectDetection.zip"},
                 'window_title': "PyeIQ - Object Detection"}

OBJ_DETECTION_GST = {'labels': "coco_labels.txt",
                     'model': "mobilenet_ssd_v2_coco_quant_postprocess.tflite",
                     'sha1': "4736e758d8d626047df7cd1b3c38c72e77fd32ee",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1kQj5Lj8fS3BUMm2NyABLo9nmS6XonjZ0",
                             'github': "https://github.com/lizeze-0515/"
                                       "pymodel/releases/download/model_3.0.0/"
                                       "eIQObjectDetectionCVGST.zip"},
                     'window_title': "PyeIQ - Object Detection OpenCV"}
                     
OBJ_COVID19 = { 'image': "demo5.jpg",
                'model_distance':"mobilenet_ssd_v2_coco_quant_postprocess.tflite",
                'model_facemask': "facemask_int8.tflite",
                'priors': "priors.txt",
                'sha1': "7a5b25a8664b7e1c6c0e8b52202d1ea0c6138858",
                'src': {'drive': "https://drive.google.com/file/d/"
                                 "1EAI_V5aueSyADzhPlPNEmEKyAX8T7ZQK",
                        'github':"https://github.com/fangxiaoying/"
                                 "pyeiq-model/releases/download/v3.0.0/"
                                 "eIQObjectDetectionCOVID19.zip"},
                'window_title': "PyeIQ - Object Detection COVID19"}
