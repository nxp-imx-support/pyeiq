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
from eiq.multimedia.utils import resize_image
from eiq.utils import args_parser, retrieve_from_id


class eIQLabelImage:
    def __init__(self):
        self.args = args_parser(image=True, label=True, model=True)
        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, "media")
        self.model_path = os.path.join(self.base_path, "model")

        self.interpreter = None
        self.image = None
        self.label = None
        self.model = None

        self.input_mean = 127.5
        self.input_std = 127.5

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def retrieve_data(self):
        retrieve_from_id(LABEL_IMAGE_MODEL_ID, self.base_path,
                         self.__class__.__name__ + ZIP, True)

        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_path, LABEL_IMAGE_MEDIA_NAME)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_path, LABEL_IMAGE_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_path, LABEL_IMAGE_MODEL_NAME)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        image = resize_image(self.interpreter.input_details,
                             self.image, use_opencv=False)
        floating_model = self.interpreter.dtype() == np.float32

        if floating_model:
            image = (np.float32(image) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()
        results = self.interpreter.get_tensor(0, squeeze=True)
        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(self.label)

        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


class eIQFireDetection:
    def __init__(self):
        self.args = args_parser(image=True, model=True)
        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, "media")
        self.model_path = os.path.join(self.base_path, "model")

        self.interpreter = None
        self.image = None
        self.model = None

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            retrieve_from_id(FIRE_DETECTION_MEDIA_ID, self.media_path,
                             FIRE_DETECTION_MEDIA_NAME)
            self.image = os.path.join(self.media_path,
                                      FIRE_DETECTION_MEDIA_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            retrieve_from_id(FIRE_DETECTION_MODEL_ID, self.model_path,
                             FIRE_DETECTION_MODEL_NAME)
            self.model = os.path.join(self.model_path,
                                      FIRE_DETECTION_MODEL_NAME)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        image = opencv.imread(self.image)
        image = resize_image(self.interpreter.input_details,
                             image, use_opencv=True)

        if self.interpreter.dtype() == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        if np.argmax(self.interpreter.get_tensor(0)) == 0:
            print("Non-Fire")
        else:
            print("Fire")
