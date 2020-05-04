import cv2 as opencv
import os
from pathlib import Path
import re
import sys

from eiq.opencv.config import *
from eiq.utils import args_parser, retrieve_from_id, retrieve_from_url


class singleShotObjectDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.args = args_parser(model=True, label=True)
        self.labels = None
        self.model_caffe = None
        self.model_proto = None
        self.nn = None

        self.base_path = os.path.join(TMP_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")
        self.media_path = os.path.join(self.base_path, "media")
        self.labeled_media_path = os.path.join(self.media_path, "labeled")

        self.blob = 0
        self.coordinates = []
        self.files = []
        self.hf = 0
        self.ir = 0
        self.ic = 0
        self.wf = 0

        self.ext = (".jpeg", ".jpg", ".png", ".bmp")
        self.normalize = 127.5
        self.opacity = 0.2
        self.scale_factor = 0.009718
        self.threshold = 0.2
        self.width = 300
        self.height = 300

    def caffeInference(self, img, img_name, nn):
        self.blob = opencv.dnn.blobFromImage(self.ir, self.scale_factor,
            (self.height, self.width),
            (self.normalize, self.normalize, self.normalize), False)

        nn.setInput(self.blob)
        det = nn.forward()
        self.ic = img.copy()
        cols = self.ir.shape[1]
        rows = self.ir.shape[0]

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]

            if (confidence > self.threshold):
                index = int(det[0, 0, i, 1])
                self.math(i, det, cols, rows)
                opencv.rectangle(self.ir,
                    (self.coordinates[0], self.coordinates[1]),
                    (self.coordinates[2], self.coordinates[3]),
                    (0, 255, 0))
                opencv.rectangle(self.ic,
                    (self.coordinates[4], self.coordinates[5]),
                    (self.coordinates[6], self.coordinates[7]),
                    (0, 255, 0), -1)

            self.coordinates = []

        opencv.addWeighted(self.ic, self.opacity, img,
            (1 - self.opacity), 0, img)

        labels = self.load_labels_from_file()

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]

            if (confidence > self.threshold):
                index = int(det[0, 0, i, 1])
                self.math(i, det, cols, rows)
                opencv.rectangle(img,
                    (self.coordinates[4], self.coordinates[5]),
                    (self.coordinates[6], self.coordinates[7]),
                    (0, 0, 0), 2)

                if (index in labels):
                    label = (labels[index] + ": " + str(confidence))
                    size, line = opencv.getTextSize(
                                    label, opencv.FONT_HERSHEY_TRIPLEX, 1, 3)
                    self.coordinates[5] = max(self.coordinates[5], size[1])
                    opencv.rectangle(img,
                        (self.coordinates[4], self.coordinates[5] - size[1]),
                        (self.coordinates[4] + size[0],
                        self.coordinates[5] + line),
                        (255, 255, 255), opencv.FILLED)
                    opencv.putText(img, label,
                        (self.coordinates[4], self.coordinates[5]),
                        opencv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

            self.coordinates = []

        self.save_labeled_image(img, img_name)

    def load_images_from_media_folder(self):
        for image in os.listdir(self.media_path):
            if (image.endswith(self.ext)):
                self.files.append(image)

    def load_labels_from_file(self):
        labels = {}

        with open(self.labels) as f:
            for line in f:
                (key, val) = line.split()
                labels[int(key)] = val

        return labels

    def math(self, i, d, c, r):
        self.coordinates.append(int(d[0, 0, i, 3] * c))
        self.coordinates.append(int(d[0, 0, i, 4] * r))
        self.coordinates.append(int(d[0, 0, i, 5] * c))
        self.coordinates.append(int(d[0, 0, i, 6] * r))
        self.coordinates.append(int(self.wf * int(d[0, 0, i, 3] * c)))
        self.coordinates.append(int(self.hf * int(d[0, 0, i, 4] * r)))
        self.coordinates.append(int(self.wf * int(d[0, 0, i, 5] * c)))
        self.coordinates.append(int(self.hf * int(d[0, 0, i, 6] * r)))

    def retrieve_data(self):
        if self.args.label is not None and os.path.isfile(self.args.label):
            self.labels = self.args.label
        else:
            self.labels = retrieve_from_id(SSD_OBJ_DETECTION_LABEL_ID,
                            self.model_path, "labels.txt")
            self.labels = os.path.join(self.labels, "labels.txt")

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model_caffe = self.args.model
        else:
            self.model_caffe = retrieve_from_url(SSD_OBJ_DETECTION_CAFFE,
                                    self.model_path)
            self.model_proto = retrieve_from_url(SSD_OBJ_DETECTION_PROTO,
                                    self.model_path)

        retrieve_from_url(SSD_OBJ_DETECTION_DEFAULT_IMAGE, self.media_path)

    def save_labeled_image(self, i, n):
        if not os.path.exists(self.labeled_media_path):
            try:
                Path(self.labeled_media_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                sys.exit("Path.mkdir(%s) function has failed"
                    % self.labeled_media_path)

        opencv.imwrite(os.path.join(self.labeled_media_path, n), i)

    def start(self):
        self.retrieve_data()
        self.load_images_from_media_folder()
        self.nn = opencv.dnn.readNetFromCaffe(self.model_proto,
                                              self.model_caffe)

    def run(self):
        self.start()

        for img_name in self.files:
            img = opencv.imread(os.path.join(self.media_path, img_name))
            self.ir = opencv.resize(img, (self.height, self.width))
            self.hf = (img.shape[0] / self.height)
            self.wf = (img.shape[1] / self.width)
            self.caffeInference(img, img_name, self.nn)
