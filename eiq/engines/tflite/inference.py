# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
from tflite_runtime.interpreter import Interpreter

from eiq.utils import Timer


class TFLiteInterpreter:
    def __init__(self, model=None):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.inference_time = None

        if model is not None:
            self.interpreter = Interpreter(model)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def dtype(self):
        return self.input_details[0]['dtype']

    def height(self):
        return self.input_details[0]['shape'][1]

    def width(self):
        return self.input_details[0]['shape'][2]

    def get_tensor(self, index, squeeze=False):
        if squeeze:
            return np.squeeze(self.interpreter.get_tensor(
                                   self.output_details[index]['index']))

        return self.interpreter.get_tensor(
                    self.output_details[index]['index'])

    def set_tensor(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

    def run_inference(self):
        timer = Timer()
        with timer.timeit():
            self.interpreter.invoke()
        self.inference_time = timer.time
