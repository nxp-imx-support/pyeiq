# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np
from PIL import Image

from config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import EMOTIONS_DETECTION, FACE_EYES_DETECTION
from eiq.modules.utils import DemoBase


class eIQEmotionsDetection(DemoBase):
    def __init__(self):
        super().__init__(download=True, image=True, video_fwk=True,
                         video_src=True, class_name=self.__class__.__name__,
                         data=EMOTIONS_DETECTION)

        self.face_cascade = None

    def gather_data(self):
        super().gather_data()

        self.face_cascade = os.path.join(self.model_dir,
                                         self.data['face_cascade'])

    @staticmethod
    def preprocess_image(image, x, y, w, h):
        image = image[y:y+h, x:x+w]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((200, 200))
        image = np.expand_dims(image, axis=0)

        if np.max(image) > 1:
            image = image / 255.0

        return image.astype(np.float32)

    def detect_emotion(self, image, x, y, w, h):
        image = self.preprocess_image(image, x, y, w, h)

        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        results = self.interpreter.get_tensor(0)
        classes = np.argmax(results, axis=1)

        if classes == 0:
            return 'anger'
        elif classes == 1:
            return 'disgust'
        elif classes == 2:
            return 'fear'
        elif classes == 3:
            return "happiness"
        elif classes == 4:
            return "neutral"
        elif classes == 5:
            return 'sadness'
        else:
            return 'surprise'

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(150, 150))

        for (x, y, w, h) in faces:
            emotion = self.detect_emotion(gray, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-5), FONT['hershey'],
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def start(self):
        self.gather_data()
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()
        self.run_inference(self.detect_face)


class eIQFaceAndEyesDetection(DemoBase):
    def __init__(self):
        super().__init__(download=True, image=True, video_fwk=True,
                         video_src=True, class_name=self.__class__.__name__,
                         data=FACE_EYES_DETECTION)

        self.eye_cascade = None
        self.face_cascade = None

    def gather_data(self):
        super().gather_data()

        self.eye_cascade = os.path.join(self.model_dir,
                                        self.data['eye_cascade'])
        self.face_cascade = os.path.join(self.model_dir,
                                         self.data['face_cascade'])

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                              (0, 255, 0), 2)

        return frame

    def start(self):
        self.gather_data()
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)

    def run(self):
        self.start()
        self.run_inference(self.detect_face)
