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
from eiq.engines.tflite import inference
from eiq.modules.classification.config import *
from eiq.multimedia.utils import gstreamer_configurations, resize_image
from eiq.utils import args_parser, retrieve_from_id, retrieve_from_url


class eIQFireDetection(object):
    def __init__(self):
        self.args = args_parser(camera=True, model=True, webcam=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, 'model')

        self.model = None
        self.video = None

        self.msg = "No Fire"
        self.msg_color = (0, 255, 0)

    def retrieve_data(self):
        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            retrieve_from_id(FIRE_DETECTION_MODEL_ID, self.model_path,
                             FIRE_DETECTION_MODEL_NAME)
            self.model = os.path.join(self.model_path,
                                      FIRE_DETECTION_MODEL_NAME)

    def detect_fire(self, image):
        img = resize_image(self.input_details, image, use_opencv=True)
        if self.input_details[0]['dtype'] == np.float32:
            img = np.array(img, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output_data)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        while True:
            ret, frame = self.video.read()
            if self.detect_fire(frame) == 0:
                self.message = "No Fire"
                self.color = (0, 255, 0)
            else:
                self.message = "Fire Detected!"
                self.color = (0, 0, 255)

            opencv.putText(frame, self.message, (50, 50),
                            opencv.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
            opencv.imshow(TITLE_FIRE_DETECTION_CAMERA, frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break
        opencv.destroyAllWindows()
