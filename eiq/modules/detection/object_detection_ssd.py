# Copyright 2018 The TensorFlow Authors
# Copyright 2021 NXP
#
# This file was copied from TensorFlow respecting its rights. All the modified
# parts below are according to TensorFlow's LICENSE terms.
#
# SPDX-License-Identifier:    Apache-2.0

import collections
import colorsys
import os
import random
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.opencv.inference import OpenCVDNN
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import OBJ_DETECTION, OBJ_DETECTION_GST
from eiq.modules.utils import DemoBase
from eiq.multimedia import gstreamer

try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False


class eIQObjectDetection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_DETECTION)

        self.colors = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * Single Shot Detection as default algorithm.\n")

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)

    def generate_colors(self):
        hsv_tuples = [(x / len(self.labels), 1., 1.)
                      for x in range(len(self.labels))]

        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255),
                                     int(x[2] * 255)), colors))
        random.seed(10101)
        random.shuffle(colors)
        random.seed(None)

        self.colors = colors

    def process_image(self, image):
        self.interpreter.set_tensor(np.expand_dims(image, axis=0))
        self.interpreter.run_inference()

        positions = self.interpreter.get_tensor(0, squeeze=True)
        classes = self.interpreter.get_tensor(1, squeeze=True)
        scores = self.interpreter.get_tensor(2, squeeze=True)

        result = []
        for idx, score in enumerate(scores):
            if score > 0.5:
                result.append({'pos': positions[idx], '_id': classes[idx]})
        return result

    def display_result(self, frame, result):
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

            label_size = cv2.getTextSize(self.labels[_id], FONT['hershey'],
                                         FONT['size'], FONT['thickness'])[0]
            label_rect_left = int(left - 3)
            label_rect_top = int(top - 3)
            label_rect_right = int(left + 3 + label_size[0])
            label_rect_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(frame, (left, top), (right, bottom),
                          self.colors[int(_id) % len(self.colors)], 6)
            cv2.rectangle(frame, (label_rect_left, label_rect_top),
                          (label_rect_right, label_rect_bottom),
                          self.colors[int(_id) % len(self.colors)], -1)
            cv2.putText(frame, self.labels[_id], (left, int(top - 4)),
                        FONT['hershey'], FONT['size'],
                        FONT['color']['black'],
                        FONT['thickness'])
            self.overlay.draw_info(frame, self.model, self.media_src,
                                   self.interpreter.inference_time,
                                   self.interpreter.input_details[0]['quantization'])

    def detect_object(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(), self.interpreter.height()))
        top_result = self.process_image(image)
        self.display_result(frame, top_result)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.labels, self.model)
        self.interpreter = TFLiteInterpreter(self.model)
        self.labels = self.load_labels(self.labels)
        self.generate_colors()

    def run(self):
        self.start()
        self.run_inference(self.detect_object)
