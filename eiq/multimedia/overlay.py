# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import cv2

from eiq.helper.config import *

class OpenCVOverlay:
    def __init__(self):
        self.frame = None
        self.time = None

    def draw_inference_time(self):
        cv2.putText(self.frame,
                    INFERENCE_TIME_MESSAGE + str(self.time),
                    (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def display_result(self, frame, time, result, labels):
        self.frame = frame
        self.time = time
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

            cv2.rectangle(self.frame, (left, top), (right, bottom),
                          self.colors[self.class_names_dict[_id]], 6)

            label_size = cv2.getTextSize(labels[_id], FONT, FONT_SIZE,
                                         FONT_THICKNESS)[0]
            label_rect_left = int(left - 3)
            label_rect_top = int(top - 3)
            label_rect_right = int(left + 3 + label_size[0])
            label_rect_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(self.frame, (label_rect_left, label_rect_top),
                          (label_rect_right, label_rect_bottom),
                          self.colors[self.class_names_dict[_id]], -1)
            cv2.putText(self.frame, labels[_id], (left, int(top - 4)),
                        FONT, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
            self.draw_inference_time()

        return self.frame
