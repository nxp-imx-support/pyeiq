# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


VIDEO_SWITCH_CORE = {'sha1': "aa8864498fed9294de27b273a2cce6a6a9097726",
                     'src': {'drive': "https://drive.google.com/file/d/"
                                      "1p3hVhEt_nFuiBw5Sj2EI7ghxmYWdggnh",
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
