# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

FIRE_CLASSIFICATION = {'image': "fire.jpg",
                       'model': "fire_detection.tflite",
                       'sha1': "2df946680459a3b20bd668f423dcdaa6b76a98b3",
                       'src': {'drive': "https://drive.google.com/file/d/"
                                        "1aDsAB2i9wl1xbkexXbiYfPawKUMYXfwH/"
                                        "view?usp=sharing",
                               'github': "https://github.com/diegohdorta/"
                                         "models/raw/master/models/"
                                         "eIQFireClassification.zip"},
                       'window_title': "PyeIQ - Fire Classification"}

OBJ_CLASSIFICATION = {'image': "cat.jpg",
                      'labels': "labels_mobilenet_quant_v1_224.txt",
                      'model': "mobilenet_v1_1.0_224_quant.tflite",
                      'sha1': "765c995e1d27c4a738f77cf13445e7b41306befc",
                      'src': {'drive': "https://drive.google.com/file/d/"
                                       "1yhQkJZwtuSyOvTOKi0ZVcIvZUR7GXv6z/"
                                       "view?usp=sharing",
                              'github': "https://github.com/diegohdorta/"
                                        "models/raw/master/models/"
                                        "eIQImageClassification.zip"},
                      'window_title': "PyeIQ - Image Classification"}

FIRE_MSG = {'fire': "Fire Detected!",
            'non-fire': "No Fire"}
