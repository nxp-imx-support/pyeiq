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
from PIL import Image

from eiq.config import BASE_DIR
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.helper.overlay import OpenCVOverlay
from eiq.modules.classification.config import *
from eiq.modules.classification.utils import load_labels
from eiq.multimedia.utils import gstreamer_configurations, resize_image
from eiq.utils import args_parser, retrieve_data


class eIQFireClassification:
    def __init__(self):
        self.args = args_parser(camera = True, camera_inference = True,
                                image=True, model=True, webcam = True)
        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, "media")
        self.model_path = os.path.join(self.base_path, "model")

        self.interpreter = None
        self.image = None
        self.model = None
        self.video = None

    def gather_data(self):
        retrieve_data(FIRE_DETECTION_MODEL_SRC, self.base_path,
                      self.__class__.__name__ + ZIP, True)
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_path,
                                      FIRE_DETECTION_MEDIA_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_path,
                                      FIRE_DETECTION_MODEL_NAME)

    def fire_classification(self, frame):
        image = resize_image(self.interpreter.input_details,
                             frame, use_opencv=True)

        if self.interpreter.dtype() == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        if np.argmax(self.interpreter.get_tensor(0)) == 0:
            opencv.putText(frame, NO_FIRE, (50, 50),
                           opencv.FONT_HERSHEY_SIMPLEX, 1, CV_GREEN, 2)
        else:
            opencv.putText(frame, FIRE, (50, 50),
                           opencv.FONT_HERSHEY_SIMPLEX, 1, CV_RED, 2)

        inference_time_overlay = OpenCVOverlay(frame, self.interpreter.inference_time)
        frame = inference_time_overlay.draw_inference_time()
        opencv.imshow(TITLE_FIRE_CLASSIFICATION, frame)

    def real_time_classification(self):
        self.video = gstreamer_configurations(self.args)

        while True:
            ret, frame = self.video.read()

            if ret:
                self.fire_classification(frame)
            else:
                print("Your video device could not capture any image.\n"\
                      "Please, check your device configurations." )
                break

            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break

        self.video.release()

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        if self.args.camera_inference:
            self.real_time_classification()
        else:
            frame = opencv.imread(self.image)
            self.fire_classification(frame)
            opencv.waitKey()

        opencv.destroyAllWindows()


class eIQObjectsClassification:
    def __init__(self):
        self.args = args_parser(camera_inference=True, image=True,
                                label=True, model=True)
        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, "media")
        self.model_path = os.path.join(self.base_path, "model")

        self.interpreter = None
        self.image = None
        self.label = None
        self.model = None

    def gather_data(self):
        retrieve_data(IMAGE_CLASSIFICATION_MODEL_SRC, self.base_path,
                         self.__class__.__name__ + ZIP, True)

        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_path, IMAGE_CLASSIFICATION_MEDIA_NAME)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_path, IMAGE_CLASSIFICATION_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_path, IMAGE_CLASSIFICATION_MODEL_NAME)

    def process_image(self, image, k=3):
        """Process an image, Return top K result in a list of 2-Tuple(confidence_score, label)"""
        input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

        self.interpreter.set_tensor(input_data)
        self.interpreter.run_inference()

        # Get outputs
        output_data = self.interpreter.get_tensor(0, squeeze=True)

        # Get top K result
        top_k = output_data.argsort()[-k:][::-1]  # Top_k index
        result = []
        for i in top_k:
            score = float(output_data[i] / 255.0)
            result.append((i, score))

        return result

    def display_result(self, top_result, frame, labels):
        """Display top K result in top right corner"""
        font = opencv.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (255, 0, 0)  # Blue color
        thickness = 1

        for idx, (i, score) in enumerate(top_result):
            x = 12
            y = 24 * idx + 24
            opencv.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                        (x, y), font, size, color, thickness)

        opencv.imshow(TITLE_IMAGE_CLASSIFICATION, frame)

    def classificate_image(self, frame):
        image = Image.fromarray(opencv.cvtColor(frame, opencv.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(),
                              self.interpreter.height()))

        top_result = self.process_image(image)
        self.display_result(top_result, frame, self.label)

    def real_time_classification(self):
        cap = opencv.VideoCapture(0)
        cap.set(opencv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(opencv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(opencv.CAP_PROP_FPS, 15)

        # Process Stream
        while True:
            ret, frame = cap.read()

            if ret:
                self.classificate_image(frame)

            if (opencv.waitKey(1) & 0xFF) == ord('q'):
                break

        cap.release()

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)
        self.label = load_labels(self.label)

    def run(self):
        self.start()

        if self.args.camera_inference:
            self.real_time_classification()
        else:
            frame = opencv.imread(self.image, opencv.IMREAD_COLOR)
            self.classificate_image(frame)
            opencv.waitKey()

        opencv.destroyAllWindows()


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

    def gather_data(self):
        retrieve_data(LABEL_IMAGE_MODEL_SRC, self.base_path,
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
        self.gather_data()
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
        labels = load_labels(self.label)

        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
