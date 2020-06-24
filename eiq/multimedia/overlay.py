# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np

from eiq.config import FONT, INF_TIME_MSG, MODEL_MSG, SRC_MSG, FPS_MSG

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

    def draw_info(self, frame, model, src, time):
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

    def display_result(self, frame, model, src, time, result, labels, colors):
        width = frame.shape[1]
        height = frame.shape[0]

        for obj in result:
            pos = obj['pos']
            _id = obj['_id']

            x1 = int(pos[1] * width)
            x2 = int(pos[3] * width)
            y1 = int(pos[0] * height)
            y2 = int(pos[2] * height)

            top = max(0, np.floor(y1 + 0.5).astype('int32'))
            left = max(0, np.floor(x1 + 0.5).astype('int32'))
            bottom = min(height, np.floor(y2 + 0.5).astype('int32'))
            right = min(width, np.floor(x2 + 0.5).astype('int32'))

            label_size = cv2.getTextSize(labels[_id], FONT['hershey'],
                                         FONT['size'], FONT['thickness'])[0]
            label_rect_left = int(left - 3)
            label_rect_top = int(top - 3)
            label_rect_right = int(left + 3 + label_size[0])
            label_rect_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(frame, (left, top), (right, bottom),
                          colors[int(_id) % len(colors)], 6)
            cv2.rectangle(frame, (label_rect_left, label_rect_top),
                          (label_rect_right, label_rect_bottom),
                          colors[int(_id) % len(colors)], -1)
            cv2.putText(frame, labels[_id], (left, int(top - 4)),
                        FONT['hershey'], FONT['size'],
                        FONT['color']['black'],
                        FONT['thickness'])
            self.draw_info(frame, model, src, time)

        return frame
