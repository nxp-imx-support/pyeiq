# Copyright 2018 The TensorFlow Authors
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import os

import cv2 as opencv
import numpy as np

from eiq.config import BASE_DIR
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.classification.config import *
from eiq.multimedia.utils import gstreamer_configurations, resize_image
from eiq.utils import args_parser, retrieve_from_id


class eIQFireDetection:
    def __init__(self):
        self.args = args_parser(camera=True, model=True, webcam=True)
        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, 'model')

        self.interpreter = None
        self.model = None
        self.video = None

    def retrieve_data(self):
        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            retrieve_from_id(FIRE_DETECTION_MODEL_ID, self.model_path,
                             FIRE_DETECTION_MODEL_NAME)
            self.model = os.path.join(self.model_path,
                                      FIRE_DETECTION_MODEL_NAME)

    def detect_fire(self, image):
        image = resize_image(self.interpreter.input_details,
                             image, use_opencv=True)

        if self.interpreter.dtype() == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        return np.argmax(self.interpreter.get_tensor(0))

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        while True:
            ret, frame = self.video.read()

            if self.detect_fire(frame) == 0:
                message = "No Fire"
                color = (0, 255, 0)
            else:
                message = "Fire Detected!"
                color = (0, 0, 255)

            opencv.putText(frame, message, (50, 50),
                           opencv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            opencv.imshow(TITLE_FIRE_DETECTION_CAMERA, frame)

            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break

        opencv.destroyAllWindows()
