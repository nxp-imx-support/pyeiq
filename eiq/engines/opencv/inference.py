# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import cv2

from eiq.utils import Timer


class OpenCVDNN:
    def __init__(self, caffe=None, proto=None):
        self.blob = None
        self.det = None
        self.net = None
        self.inference_time = None

        if caffe and proto:
            self.net = cv2.dnn.readNetFromCaffe(proto, caffe)

    def create_blob(self, image, dims, scale, normalize):
        self.blob = cv2.dnn.blobFromImage(image, scale, (dims, dims),
                                          (normalize, normalize,
                                           normalize), False)

    def run_inference(self):
        timer = Timer()
        self.net.setInput(self.blob)

        with timer.timeit():
            self.det = self.net.forward()

        self.inference_time = timer.time
