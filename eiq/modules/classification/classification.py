# Copyright 2018 The TensorFlow Authors
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import os

import cv2
import numpy as np
from PIL import Image

from eiq.config import BASE_DIR
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.multimedia.overlay import OpenCVOverlay
from eiq.modules.classification.config import *
from eiq.modules.classification.utils import load_labels
from eiq.modules.utils import real_time_inference
from eiq.multimedia.utils import gstreamer_configurations, resize_image
from eiq.utils import args_parser, Downloader


class eIQFireClassification:
    def __init__(self):
        self.args = args_parser(download=True,image=True,
                                model=True, video_fwk=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.interpreter = None
        self.image = None
        self.model = None
        self.overlay = OpenCVOverlay()

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(FIRE_DETECTION_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               FIRE_DETECTION_MODEL_SHA1, True)

        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      FIRE_DETECTION_MEDIA_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_dir,
                                      FIRE_DETECTION_MODEL_NAME)

    def fire_classification(self, frame):
        image = resize_image(self.interpreter.input_details,
                             frame, use_opencv=True)

        if self.interpreter.dtype() == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        if np.argmax(self.interpreter.get_tensor(0)) == 0:
            cv2.putText(frame, NO_FIRE, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, CV_GREEN, 2)
        else:
            cv2.putText(frame, FIRE, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, CV_RED, 2)

        self.overlay.draw_inference_time(frame, self.interpreter.inference_time)
        cv2.imshow(TITLE_FIRE_CLASSIFICATION, frame)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        if self.args.video_src:
            real_time_inference(inference_func=self.fire_classification,
                                args=self.args)
        else:
            frame = cv2.imread(self.image)
            self.fire_classification(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()


class eIQObjectsClassification:
    def __init__(self):
        self.args = args_parser(download=True, image=True, label=True,
                                model=True, video_fwk=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.interpreter = None
        self.image = None
        self.label = None
        self.model = None
        self.overlay = OpenCVOverlay()

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.8
        self.font_color = (0, 127, 255)
        self.font_thickness = 2

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_CLASSIFICATION_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               OBJ_CLASSIFICATION_MODEL_SHA1, True)

        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir, OBJ_CLASSIFICATION_MEDIA_NAME)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_dir, OBJ_CLASSIFICATION_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_dir, OBJ_CLASSIFICATION_MODEL_NAME)

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
            x = 20
            y = 35 * idx + 35
            cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                           (x, y), self.font, self.font_size,
                           self.font_color, self.font_thickness)

        self.overlay.draw_inference_time(frame, self.interpreter.inference_time)
        cv2.imshow(TITLE_OBJ_CLASSIFICATION, frame)

    def classificate_image(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(),
                              self.interpreter.height()))

        top_result = self.process_image(image)
        self.display_result(top_result, frame, self.label)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)
        self.label = load_labels(self.label)

    def run(self):
        self.start()
        if self.args.video_src:
            real_time_inference(inference_func=self.classificate_image,
                                args=self.args)
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.classificate_image(frame)
            cv2.waitKey()
        cv2.destroyAllWindows()
