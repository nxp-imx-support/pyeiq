# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


VIDEO_SWITCH_CORE = {'sha1': "89effa43394b8dd9949ce422224bf3a1ad8f56b6",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "13tUttASMoIombyMCkrFf_Gl64A5Qt30-/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "eIQVideoSwitchCore.zip"},
                     'window_title': "PyeIQ - Object Detection Switch Cores"}

CPU_IMG = os.path.join("/tmp", "cpu.jpg")
NPU_IMG = os.path.join("/tmp", "npu.jpg")


JPEG_EOF = b'\xff\xd9'

CPU = 0
NPU = 1

PAUSE = "kill -STOP {}"
RESUME = "kill -CONT {}"
RUN = "{} -a {}"
