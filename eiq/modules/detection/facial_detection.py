# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import FACE_EYES_DETECTION
from eiq.modules.utils import DemoBase
from eiq.utils import Timer


class eIQFaceAndEyesDetection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, FACE_EYES_DETECTION)
        self.timer = Timer()

        self.eye_cascade = None
        self.face_cascade = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * OpenCV as inference engine."
                "\n   * Haar Cascade as default algorithm.\n")

    def usage(self, name=None, labels=False, model=False):
        super().usage(name=name, labels=labels, model=model)

    def gather_data(self):
        super().gather_data()

        self.eye_cascade = os.path.join(self.model_dir,
                                        self.data['eye_cascade'])
        self.face_cascade = os.path.join(self.model_dir,
                                         self.data['face_cascade'])

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with self.timer.timeit():
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                              (0, 255, 0), 2)

        self.overlay.draw_info(frame, "haarcascade_frontalface_default.xml",
                               self.media_src, self.timer.time)

        return frame

    def start(self):
        self.gather_data()
        self.validate_data(self.image, self.model,
                           self.eye_cascade, self.face_cascade)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)

    def run(self):
        self.start()
        self.run_inference(self.detect_face)
