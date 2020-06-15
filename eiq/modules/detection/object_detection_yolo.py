# Copyright 2020 Michael Kuo
# Copyright 2020 NXP Semiconductors
#
# Parts of this file were copied from Michael Kuo's github repository 
# respecting its rights. All the modified parts below are according to
# MIT LICENSE terms.
#
# SPDX-License-Identifier:    MIT

import os

import cv2
import numpy as np
from PIL import Image

from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import *
from eiq.modules.utils import DemoBase


class eIQObjectsDetectionYOLOV3(DemoBase):
    def __init__(self):
        super().__init__(download=True, image=True, labels=True,
                         model=True, video_fwk=True, video_src=True,
                         class_name=self.__class__.__name__,
                         data=OBJ_DETECTION_YOLOV3)

        self.anchors = [[0.57273, 0.677385], [1.87446, 2.06253],
                        [3.33843, 5.47434], [7.88282, 3.52778],
                        [9.77052, 9.16828]]
        self.block_size = 32
        self.grid_height = 13
        self.grid_width = 13
        self.boxes_per_block = 5

        self.overlap_threshold = 0.2
        self.threshold = 0.3

        self.left = 0
        self.top = 1
        self.right = 2
        self.bottom = 3
        self.confidence = 4
        self.classes = 5

    @staticmethod
    def expit(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def sort_results(self, e):
        return e[self.confidence]

    def get_input_data(self, image):
        image = image.resize((self.interpreter.width(),
                              self.interpreter.height()))

        n = np.array(image, dtype='float32')
        n = n / 255.0

        return np.array([n], dtype='float32')

    @staticmethod
    def load_labels(labels_path):
        labels = []

        with open(labels_path, 'r') as lines:
            for line in lines.readlines():
                labels.append(line.rstrip())

        return labels

    def non_maximal_suppression(self, results):
        predictions = []

        if len(results):
            results.sort(reverse=True, key=self.sort_results)
            best_prediction = results.pop(0)
            predictions.append(best_prediction)

            while len(results) != 0:
                prediction = results.pop(0)
                overlaps = False

                for j in range(len(predictions)):
                    previous_prediction = predictions[j]

                    intersect_proportion = 0.0
                    primary = previous_prediction
                    secondary = prediction

                    if (primary[self.left] < secondary[self.right]) \
                            and (primary[self.right] > secondary[self.left]) \
                            and (primary[self.top] < secondary[self.bottom]) \
                            and (primary[self.bottom] > secondary[self.top]):
                        intersection = max(0, min(primary[self.right],
                                                  secondary[self.right]) - max(primary[self.left],
                                                                               secondary[self.left])) \
                                       * max(0, min(primary[self.bottom],
                                                    secondary[self.bottom]) - max(primary[self.top],
                                                                                  secondary[self.top]))

                        main = np.abs(primary[self.right] - primary[self.left]) \
                               * np.abs(primary[self.bottom] - primary[self.top])
                        intersect_proportion = intersection / main

                    overlaps = overlaps or (intersect_proportion > self.overlap_threshold)

                if not overlaps:
                    predictions.append(prediction)

        return predictions

    def check_result(self, data):
        results = []

        for row in range(self.grid_height):
            for column in range(self.grid_width):
                for box in range(self.boxes_per_block):
                    item = data[row][column]
                    offset = (len(self.labels) + 5) * box

                    confidence = self.expit(item[offset + 4])

                    classes = item[offset + 5: offset + 5 + len(self.labels)]
                    classes = self.softmax(classes)

                    detected_class = np.argmax(classes)
                    max_class = classes[detected_class]

                    confidence_in_class = max_class * confidence

                    if confidence_in_class >= self.threshold:
                        x_pos = (column + self.expit(item[offset])) \
                                * self.block_size
                        y_pos = (row + self.expit(item[offset + 1])) \
                                * self.block_size
                        w = (np.exp(item[offset + 2]) * self.anchors[box][0]) \
                            * self.block_size
                        h = (np.exp(item[offset + 3]) * self.anchors[box][1]) \
                            * self.block_size

                        left = max(0, x_pos - w / 2)
                        top = max(0, y_pos - h / 2)
                        right = min(self.interpreter.width() - 1,
                                    x_pos + w / 2)
                        bottom = min(self.interpreter.height() - 1,
                                     y_pos + h / 2)

                        results.append([left, top, right, bottom,
                                        confidence_in_class,
                                        self.labels[detected_class]])

        return self.non_maximal_suppression(results)

    def detect_objects(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.get_input_data(image)
        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()
        data = self.interpreter.get_tensor(0)[0]
        self.draw_rectangles(frame, self.check_result(data))

        return frame

    def draw_rectangles(self, frame, predictions):
        width = frame.shape[1]
        height = frame.shape[0]
        width_ratio = width / self.interpreter.width()
        height_ratio = height / self.interpreter.height()

        for element in predictions:
            x1 = int(element[self.left] * width_ratio)
            x2 = int(element[self.right] * width_ratio)
            y1 = int(element[self.top] * height_ratio)
            y2 = int(element[self.bottom] * height_ratio)

            top = int(max(0, np.floor(y1 + 0.5).astype('int32')))
            left = int(max(0, np.floor(x1 + 0.5).astype('int32')))
            bottom = int(min(height, np.floor(y2 + 0.5).astype('int32')))
            right = int(min(width, np.floor(x2 + 0.5).astype('int32')))

            label_size = cv2.getTextSize(element[5], FONT, FONT_SIZE,
                                         FONT_THICKNESS)[0]
            label_left = int(left - 3)
            label_top = int(top - 3)
            label_right = int(left + 3 + label_size[0])
            label_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.rectangle(frame, (label_left, label_top),
                          (label_right, label_bottom),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, element[self.classes], (left, top - 4),
                        FONT, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

    def start(self):
        self.gather_data()
        self.labels = self.load_labels(self.labels)
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()
        self.run_inference(self.detect_objects)
