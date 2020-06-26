# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

VIDEO_SWITCH_CORE = {'sha1': "a0ddb9d9874282e0ff4e4b063c8688d55eecb4eb",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1-GJ-672OC_CmyNTzT1C4OLFrtRti5pAC/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "eIQVideoSwitchCore.zip"},
                     'window_title': "PyeIQ - Object Detection Switch Cores"}

JPEG_EOF = b'\xff\xd9'

CPU = 0
NPU = 1

PAUSE = "kill -STOP {}"
RESUME = "kill -CONT {}"
RUN = "{} -a {}"
