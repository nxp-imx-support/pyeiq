# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

# Switch Label Image

SWITCH_IMAGE = {'bin': "label_image",
                'labels': "labels.txt",
                'model': "mobilenet_v1_1.0_224_quant.tflite",
                'sha1': "2e71d5c1ba5695713260a0492b971c84a0785213",
                'src': {'drive': "https://drive.google.com/file/d/"
                                 "16RU5SHg72S1C-arbvA2nYlwAYTyKBYjz/"
                                 "view?usp=sharing",
                        'github': "https://github.com/diegohdorta/"
                                  "models/raw/master/models/"
                                  "eIQSwitchLabelImage.zip"},
                'msg': {'confidence': "<b>CONFIDENCE</b>",
                        'inf_time': "<b>INFERENCE TIME</b>",
                        'labels': "<b>LABELS</b>",
                        'model_name': "<b>MODEL NAME</b>",
                        'select_img': "<b>SELECT AN IMAGE</b>"},
                'window_title': "PyeIQ - Label Image Switch App"}

RUN_LABEL_IMAGE = "VSI_NN_LOG_LEVEL=0 {0} -m {1} -t 1 -i {2} -l {3} -a {4} -v 0 -c 100"

REGEX_GET_INTEGER_FLOAT = "\d+\.\d+|\d+"
REGEX_GET_STRING = "[^a-zA-Z\s]"
