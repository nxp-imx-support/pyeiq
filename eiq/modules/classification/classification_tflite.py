# Copyright 2018 The TensorFlow Authors
#
## Copyright 2021 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.classification.config import OBJ_CLASSIFICATION
from eiq.modules.utils import DemoBase

class eIQObjectClassificationTFLite(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_CLASSIFICATION)

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * CNN as default algorithm.\n")

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)

    def load_labels(self, label_path):
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def process_image(self, image, k=3):
        input_data = np.expand_dims(image, axis=0)
        self.interpreter.set_tensor(input_data)
        self.interpreter.run_inference()
        output_data = self.interpreter.get_tensor(0, squeeze=True)

        top_k = output_data.argsort()[-k:][::-1]
        result = []
        for i in top_k:
            score = float(output_data[i] / 255.0)
            result.append((i, score))
        return result

    def display_result(self, top_result, frame, labels):
        for idx, (i, score) in enumerate(top_result):
            x = 3
            y = 35 * idx + 60
            cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                        (x, y), FONT['hershey'], FONT['size'],
                        FONT['color']['black'], FONT['thickness'] + 2)
            cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                        (x, y), FONT['hershey'], FONT['size'],
                        FONT['color']['orange'], FONT['thickness'])

        self.overlay.draw_info(frame, self.model, self.media_src,
                               self.interpreter.inference_time,
                               self.interpreter.input_details[0]['quantization'])

    def classify_image(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(),
                              self.interpreter.height()))

        top_result = self.process_image(image)
        self.display_result(top_result, frame, self.labels)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.labels, self.model)
        self.interpreter = TFLiteInterpreter(self.model)
        self.labels = self.load_labels(self.labels)

    def run(self):
        self.start()
        self.run_inference(self.classify_image)
