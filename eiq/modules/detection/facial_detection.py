# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import numpy as np
import cv2

from eiq.config import BASE_DIR
from eiq.modules.detection.config import *
from eiq.multimedia.utils import gstreamer_configurations
from eiq.utils import args_parser, Downloader


class eIQFaceAndEyesDetection:
    def __init__(self):
        self.args = args_parser(download=True, image=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.eye_cascade = None
        self.face_cascade = None

        self.image = None
        self.video = None

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

    def real_time_detection(self):
        self.video = gstreamer_configurations(self.args)
        if (not self.video) or (not self.video.isOpened()):
            sys.exit("Your video device could not be found. Exiting...")

        while True:
            ret, frame = self.video.read()
            if ret:
                self.detect_face(frame)
            else:
                print("Your video device could not capture any image.\n"\
                      "Please, check your device's configurations." )
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        self.video.release()

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade)
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade)

    def run(self):
        self.start()

        if self.args.video_src:
            self.real_time_detection()
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_face(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()
