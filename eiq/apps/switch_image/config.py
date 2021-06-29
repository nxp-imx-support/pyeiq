# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

# Switch Label Image

LIB_PATH = "/usr/lib"
LIBNN = "libneuralnetworks.so"

SWITCH_IMAGE = {'bin': "label_image",
                'labels': "labels.txt",
                'model': "mobilenet_v1_1.0_224_quant.tflite",
                'sha1': "6cd4ca55c0beb56bcd7f053d2f37cc612f58c710",
                'src': {'drive': "https://drive.google.com/file/d/"
                                 "1omIHVWVR06evXYG1pLII1kodElFujfFR",
                        'github': "https://github.com/KaixinDing/"
                                  "pyeiq_model/releases/download/3.0.0/"
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
