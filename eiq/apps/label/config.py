# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

IMAGES_DRIVE_ID = "12wRWlzYQp1ABp2hVuZcPiO5FipEy6HIJ"
IMAGES_DRIVE_NAME = "bmp_examples"

DEFAULT_TFLITE_IMAGE = "/usr/bin/tensorflow-lite-2.1.0/examples/grace_hopper.bmp"

TFLITE_LABEL_IMAGE_ACCEL = "cd /usr/bin/tensorflow-lite-2.1.0/examples/; " \
                           "VSI_NN_LOG_LEVEL=0 ./label_image " \
                           "-m /usr/bin/tensorflow-lite-2.1.0/examples/mobilenet_v1_1.0_224_quant.tflite " \
                           "-t 1 -i {} " \
                           "-l labels.txt -a 1 -v 0 -c 100"

TFLITE_LABEL_IMAGE_NO_ACCEL = "cd /usr/bin/tensorflow-lite-2.1.0/examples/; " \
                              "VSI_NN_LOG_LEVEL=0 ./label_image " \
                              "-m /usr/bin/tensorflow-lite-2.1.0/examples/mobilenet_v1_1.0_224_quant.tflite " \
                              "-t 1 -i {} " \
                              "-l labels.txt -a 0 -v 0 -c 100"

REGEX_GET_INTEGER_FLOAT = "\d+\.\d+|\d+"

REGEX_GET_STRING = "[^a-zA-Z\s]"

