# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import cv2
import numpy as np
from PIL import Image

from eiq.config import BASE_DIR
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import *
from eiq.modules.utils import real_time_inference
from eiq.multimedia.utils import gstreamer_configurations
from eiq.utils import args_parser, Downloader


class eIQEmotionsDetection:
    def __init__(self):
        self.args = args_parser(download=True, image=True,
                                video_fwk=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.face_cascade = None
        self.model = None
        self.interpreter = None

        self.image = None

    def gater_data(self):
        download = Downloader(self.args)
        download.retrieve_data(EMOTIONS_DETECTION_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               EMOTIONS_DETECTION_SHA1, True)
        self.face_cascade = os.path.join(self.model_dir,
                                        EMOTIONS_DETECTION_CASCADE_FACE_NAME)
        self.model = os.path.join(self.model_dir,
                                  EMOTIONS_DETECTION_MODEL_NAME)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      FACE_EYES_DETECTION_MEDIA_NAME)

    def preprocess_image(self, image, x, y, w, h):
        image = image[y:y+h, x:x+w]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((200,200))
        image = np.expand_dims(image, axis=0)

        if(np.max(image) > 1):
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
            cv2.putText(frame, emotion, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow(TITLE_EMOTIONS_DETECTION, frame)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gater_data()
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)
        self.interpreter = TFLiteInterpreter(self.model)

    def run(self):
        self.start()

        if self.args.video_src:
            real_time_inference(self.detect_face, self.args)
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_face(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()


class eIQFaceAndEyesDetection:
    def __init__(self):
        self.args = args_parser(download=True, image=True,
                                video_fwk=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.eye_cascade = None
        self.face_cascade = None

        self.image = None

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(FACE_EYES_DETECTION_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               FACE_EYES_DETECTION_SHA1, True)
        self.eye_cascade = os.path.join(self.model_dir,
                                        FACE_EYES_DETECTION_CASCADE_EYES_NAME)
        self.face_cascade = os.path.join(self.model_dir,
                                         FACE_EYES_DETECTION_CASCADE_FACE_NAME)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      FACE_EYES_DETECTION_MEDIA_NAME)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                              (0, 255, 0), 2)

        cv2.imshow(TITLE_FACE_EYES_DETECTION, frame)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)

    def run(self):
        self.start()

        if self.args.video_src:
            real_time_inference(self.detect_face, self.args)
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_face(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()
