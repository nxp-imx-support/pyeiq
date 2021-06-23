# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


VIDEO_SWITCH_CORE = {'sha1': "94909aaa1405c8b05fe04bc4b736939a6bbb373a",
                     'src': {'drive': "https://drive.google.com/file/d/''
                                      "1TMVuxKlzNwzRXZOhKc2nalMP5TiZ3nCU/"
                                      "view?usp=sharing",
                             'github': "https://github.com/KaixinDing/"
                                       "pyeiq_model/releases/download/3.0.0/"
                                       "eIQVideoSwitchCore.zip"},
                     'window_title': "PyeIQ - Object Detection Switch Cores"}

CPU_IMG = os.path.join("/tmp", "cpu.jpg")
GPU_IMG = os.path.join("/tmp", "gpu.jpg")
NPU_IMG = os.path.join("/tmp", "npu.jpg")

JPEG_EOF = b'\xff\xd9'

CPU = 0
GPU = 1
NPU = 2

PAUSE = "kill -STOP {}"
RESUME = "kill -CONT {}"
RUN = "USE_GPU_INFERENCE={} {} -a {}"
