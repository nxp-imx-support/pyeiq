# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np

from eiq.config import FONT, INF_TIME_MSG, MODEL_MSG, QUANT_MSG0, QUANT_MSG1, SRC_MSG, FPS_MSG

class OpenCVOverlay:
    def __init__(self):
        self.time = None
        self.fps = None

    def draw_fps(self, frame, fps):
        fps_msg = "{}: {}".format(FPS_MSG, fps)
        x_offset = frame.shape[1] - (cv2.getTextSize(fps_msg, FONT['hershey'],
                                                     0.8, 2)[0][0] + 10)
        cv2.putText(frame, fps_msg,
                    (x_offset, 25), FONT['hershey'], 0.8,
                    FONT['color']['black'], 2, cv2.LINE_AA)
        cv2.putText(frame, fps_msg,
                    (x_offset, 25), FONT['hershey'], 0.8,
                    FONT['color']['white'], 1, cv2.LINE_AA)

    def draw_info(self, frame, model, src, time, quant=None):
        model = os.path.basename(model)

        cv2.putText(frame, "{}: {}".format(INF_TIME_MSG, time),
                    (3, 20), FONT['hershey'], 0.5,
                    FONT['color']['black'], 2, cv2.LINE_AA)
        cv2.putText(frame, "{}: {}".format(INF_TIME_MSG, time),
                    (3, 20), FONT['hershey'], 0.5,
                    FONT['color']['white'], 1, cv2.LINE_AA)

        y_offset = frame.shape[0] - cv2.getTextSize(src, FONT['hershey'],
                                                    0.5, 2)[0][1]
        cv2.putText(frame, "{}: {}".format(SRC_MSG, src),
                    (3, y_offset), FONT['hershey'], 0.5,
                    FONT['color']['black'], 2, cv2.LINE_AA)
        cv2.putText(frame, "{}: {}".format(SRC_MSG, src),
                    (3, y_offset), FONT['hershey'], 0.5,
                    FONT['color']['white'], 1, cv2.LINE_AA)

        y_offset -= (cv2.getTextSize(model, FONT['hershey'], 0.5, 2)[0][1] + 3)
        cv2.putText(frame, "{}: {}".format(MODEL_MSG, model),
                    (3, y_offset), FONT['hershey'], 0.5,
                    FONT['color']['black'], 2, cv2.LINE_AA)
        cv2.putText(frame, "{}: {}".format(MODEL_MSG, model),
                    (3, y_offset), FONT['hershey'], 0.5,
                    FONT['color']['white'], 1, cv2.LINE_AA)

        if not quant or (quant[0] == 0):
            y_offset -= (cv2.getTextSize(QUANT_MSG1, FONT['hershey'], 0.6, 2)[0][1] + 3) + 5
            cv2.putText(frame, QUANT_MSG1, (3, y_offset), FONT['hershey'], 0.6,
                        FONT['color']['black'], 2, cv2.LINE_AA)
            cv2.putText(frame, QUANT_MSG1, (3, y_offset), FONT['hershey'], 0.6,
                        FONT['color']['red'], 1, cv2.LINE_AA)

            y_offset -= (cv2.getTextSize(QUANT_MSG0, FONT['hershey'], 0.6, 2)[0][1] + 3)
            cv2.putText(frame, QUANT_MSG0, (3, y_offset), FONT['hershey'], 0.6,
                        FONT['color']['black'], 2, cv2.LINE_AA)
            cv2.putText(frame, QUANT_MSG0, (3, y_offset), FONT['hershey'], 0.6,
                        FONT['color']['red'], 1, cv2.LINE_AA)
