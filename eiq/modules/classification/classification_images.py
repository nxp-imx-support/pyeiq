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
from eiq.multimedia.utils import resize_image
from eiq.utils import args_parser, retrieve_from_id, retrieve_from_url


class eIQLabelImage(object):
    def __init__(self, **kwargs):
        self.args = args_parser(image=True, label=True, model=True)
        self.__dict__.update(kwargs)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = {'image': LABEL_IMAGE_DEFAULT_IMAGE,
                         'labels': LABEL_IMAGE_LABELS,
                         'model': LABEL_IMAGE_MODEL
                         }

        self.image = None
        self.label = None
        self.model = None

        self.input_mean = 127.5
        self.input_std = 127.5

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(self.to_fetch['image'], self.name)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = retrieve_from_url(self.to_fetch['labels'], self.name)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_url(self.to_fetch['model'], self.name)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        image = resize_image(self.input_details, self.image, use_opencv=False)
        floating_model = self.input_details[0]['dtype'] == np.float32

        if floating_model:
            image = (np.float32(image) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(self.label)

        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


class eIQFireDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(image=True, model=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, 'media')
        self.model_path = os.path.join(self.base_path, 'model')

        self.image = None
        self.model = None

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(FIRE_DETECTION_DEFAULT_IMAGE,
                                           self.media_path)

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
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        image = opencv.imread(self.image)
        image = resize_image(self.input_details, image, use_opencv=True)

        if self.input_details[0]['dtype'] == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if np.argmax(output_data) == 0:
            print("Non-Fire")
        else:
            print("Fire")
