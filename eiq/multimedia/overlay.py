# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import cv2 as opencv

from eiq.helper.config import *

class OpenCVOverlay:
    def __init__(self, overlay_frame, time):
        self.frame = overlay_frame
        self.time = time
        
    def draw_inference_time(self):
        opencv.putText(self.frame,
                       INFERENCE_TIME_MESSAGE + str(self.time),
                       (3, 12),
                       opencv.FONT_HERSHEY_SIMPLEX,
                       0.4,
                       (255, 255, 255), 1, opencv.LINE_AA)
        return self.frame
