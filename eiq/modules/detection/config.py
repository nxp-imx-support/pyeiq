# Copyright 2020 NXP Semiconductors
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
                                        "1HPWV4W4FnrfG14tnZF7bugkgoCmMpM-p/"
                                        "view?usp=sharing",
                               'github': "https://github.com/diegohdorta/"
                                         "models/raw/master/models/"
                                         "eIQFaceAndEyesDetection.zip"},
                       'window_title': "PyeIQ - Face and Eyes Detection"}

FACIAL_EXPRESSION_DETECTION = {'expressions': ['anger', 'disgust', 'fear', 'happiness',
                                               'neutral', 'sadness'],
                               'face_cascade': "haarcascade_frontalface_default.xml",
                               'image': "grace_hopper.jpg",
                               'model': "model.tflite",
                               'sha1': "d112d8e378f8457a48b66aea80bc4e714e8e2e41",
                               'src': {'drive': "https://drive.google.com/file/d/"
                                                "15VTXVJ7GwS_Rr2vvl9xve9UAcFO3LFzG/"
                                                "view?usp=sharing",
                                       'github': "https://github.com/diegohdorta/"
                                                 "models/raw/master/models/"
                                                 "eIQEmotionsDetection.zip"},
                               'window_title': "PyeIQ - Facial Expression Detection"}

OBJ_DETECTION = {'image': "bus.jpg",
                 'labels': "coco_labels.txt",
                 'model': "detect.tflite",
                 'sha1': "73b8bb0749f275c10366553bab6f5f313230c527",
                 'src': {'drive': "https://drive.google.com/file/d/"
                                  "1xdjdKPOH2PFPStbU2K2TEuFJN8Las3ag/"
                                  "view?usp=sharing",
                         'github': "https://github.com/diegohdorta/"
                                   "models/raw/master/models/"
                                   "eIQObjectDetection.zip"},
                 'window_title': "PyeIQ - Object Detection"}

OBJ_DETECTION_DNN = {'caffe': "MobileNetSSD_deploy.caffemodel",
                     'config': {'dims': 300,
                                'normalize': 127.5,
                                'scale': 0.009718,
                                'threshold': 0.2},
                     'image': "dog.jpg",
                     'labels': "labels.txt",
                     'proto': "MobileNetSSD_deploy.prototxt",
                     'sha1': "f9894307c83f8ddec91af76b8cd6f3dc07196dc0",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1_qeq3CxK-xhrVX4qdsmnWQ_dwlMCWF76/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "object_detection_image.zip"},
                     'window_title': "PyeIQ - Object Detection DNN"}

OBJ_DETECTION_GST = {'labels': "coco_labels.txt",
                     'model': "mobilenet_ssd_v2_coco_quant_postprocess.tflite",
                     'sha1': "4736e758d8d626047df7cd1b3c38c72e77fd32ee",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1l2gKkPIPPPI0Q4QrVBXgSOkQUDVHogOi/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "eIQObjectDetectionCVGST.zip"},
                     'window_title': "PyeIQ - Object Detection OpenCV"}

OBJ_DETECTION_SSD = {'image': "dog.jpg",
                     'labels': "coco_classes.txt",
                     'model': "ssd_mobilenet_v2.tflite",
                     'sha1': "fadfdb7c4bf056edee09cd37c87d06bb19e6ef83",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1t3VmNdkpfp4M-jyz_AD2QUeyS78FyqQT/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "object_detection_ssd.zip"},
                     'window_title': "PyeIQ - Object Detection SSD"}

OBJ_DETECTION_YOLOV3 = {'config': {'anchors': [[0.57273, 0.677385],
                                               [1.87446, 2.06253],
                                               [3.33843, 5.47434],
                                               [7.88282, 3.52778],
                                               [9.77052, 9.16828]],
                                   'block_size': 32,
                                   'boxes': 5,
                                   'grid_h': 13,
                                   'grid_w': 13,
                                   'overlap_threshold': 0.2,
                                   'threshold': 0.3},
                        'image': "example.jpg",
                        'labels': "labels.txt",
                        'model': "tiny_yolov3.tflite",
                        'sha1': "406438b9a5a530f6f6874341219a749e4f209b6e",
                        'src': {'drive': "https://drive.google.com/file/d/"
                                         "1gzUVbyDZrgAFDyZG3aRpnsClufL9Q7Hx/"
                                         "view?usp=sharing",
                                'github': "https://github.com/diegohdorta/"
                                          "models/raw/master/models/"
                                          "eIQObjectDetectionYOLOV3.zip"},
                        'window_title': "PyeIQ - Object Detection YOLOV3"}

POSE_DETECTION = {'config': {'mean': 127.5,
                             'std': 127.5,
                             'confidence': 0.4},
                  'image': "example.jpg",
                  'model': "posenet_mobilenet_v1_100_257x257"
                           "_multi_kpt_stripped.tflite",
                  'sha1': "fd4fa2ebdaeb43a8f6633888456bfb068e04a5ae",
                  'src': {'drive': "https://drive.google.com/file/d/"
                                   "1EoAGgsooOQbPYV7Okgt6VbcrdMEHZRkW/"
                                   "view?usp=sharing",
                          'github': "https://github.com/diegohdorta/models/"
                                    "raw/master/models/posenet_model.zip"},
                  'window_title': "PyeIQ - Pose Detection"}