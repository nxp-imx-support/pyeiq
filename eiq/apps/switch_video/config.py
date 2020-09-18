# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


VIDEO_SWITCH_CORE = {'sha1': "11ef7975a5e60d4cafb821fb474f314c760a04cf",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1VPBgzeONHbUaQudXJB5GNOp5jCwiX7Hx/"
                                      "view?usp=sharing",
                             'github': "https://github.com/diegohdorta/"
                                       "models/raw/master/models/"
                                       "eIQVideoSwitchCorev2.zip"},
                     'window_title': "PyeIQ - Object Detection Switch Cores"}

CPU_IMG = os.path.join("/tmp", "cpu.jpg")
NPU_IMG = os.path.join("/tmp", "npu.jpg")


JPEG_EOF = b'\xff\xd9'

CPU = 0
NPU = 1

PAUSE = "kill -STOP {}"
RESUME = "kill -CONT {}"
RUN = "{} -a {}"
