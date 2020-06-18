# Copyright 2018 The TensorFlow Authors
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.armnn.inference import aNNInterpreter
from eiq.modules.classification.config import FIRE_CLASSIFICATION, FIRE_MSG, OBJ_CLASSIFICATION
from eiq.modules.utils import DemoBase


class eIQFireClassification(DemoBase):
    def __init__(self):
        super().__init__(download=True, image=True, model=True,
                         video_fwk=True, video_src=True,
                         class_name=self.__class__.__name__,
                         data=FIRE_CLASSIFICATION)

    def fire_classification(self, frame):
        image = cv2.resize(frame, (128,
                                   128))
        image = np.expand_dims(image, axis=0)

        image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        if np.argmax(self.interpreter.get_tensor(0)) == 0:
            cv2.putText(frame, FIRE_MSG['non-fire'], (3, 50),
                        FONT['hershey'], 1, FONT['color']['green'], 2)
        else:
            cv2.putText(frame, FIRE_MSG['fire'], (3, 50),
                        FONT['hershey'], 1, FONT['color']['red'], 2)

        self.overlay.draw_inference_time(frame, self.interpreter.inference_time)

        return frame

    def start(self):
        self.gather_data()
        self.interpreter = aNNInterpreter(self.model)

    def run(self):
        self.start()
        self.run_inference(self.fire_classification)


class eIQObjectsClassification(DemoBase):
    def __init__(self):
        super().__init__(download=True, image=True, labels=True,
                         model=True, video_fwk=True, video_src=True,
                         class_name=self.__class__.__name__,
                         data=OBJ_CLASSIFICATION)

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
            y = 35 * idx + 35
            cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                        (x, y), FONT['hershey'], FONT['size'],
                        FONT['color']['orange'], FONT['thickness'])

        self.overlay.draw_inference_time(frame, self.interpreter.inference_time)

    def classify_image(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((128,
                              128))

        top_result = self.process_image(image)
        self.display_result(top_result, frame, self.labels)

        return frame

    def start(self):
        self.gather_data()
        self.interpreter = aNNInterpreter(self.model)
        self.labels = self.load_labels(self.labels)

    def run(self):
        self.start()
        self.run_inference(self.classify_image)