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
                     
OBJ_COVID19 = { 'image': "demo5.jpg",
                'model_distance':"mobilenet_ssd_v2_coco_quant_postprocess.tflite",
                'model_facemask': "facemask_int8.tflite",
                'priors': "priors.txt",
                'sha1': "42bb0f476aa98b2e901c8300965a3b0a04deb87c",
                'src': {'drive': "https://drive.google.com/file/d/"
                                 "1799w8eIbbbo8nBeQLFHfa7RDCUb8XG5g/"
                                 "view?usp=sharing",
                        'github':"https://github.com/fangxiaoying/"
                                 "pyeiq-model/releases/download/v3.0.0/"
                                 "eIQObjectDetectionCOVID19.zip"},
                'window_title': "PyeIQ - Object Detection COVID19"}