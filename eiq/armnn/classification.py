# Copyright 2020 ARM Software
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from ARM respecting its rights. All the modified
## parts below are according to ARM's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0
##
## References:
## https://github.com/ARM-software/ML-examples/tree/master/pyarmnn-fire-detection

import argparse
from pkg_resources import parse_version
from timeit import default_timer as timer

import cv2
import numpy as np
import pyarmnn as ann
from pyarmnn import __version__ as pyarmnn_version

from eiq.armnn import config

assert parse_version(pyarmnn_version) >= parse_version('19.11.0'), \
    "This demo requires pyarmnn version >= 19.11.0"


class eIQFireDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(image=True, model=True)
        self.name = self.__class__.__name__
        self.to_fetch = {'image': config.FIRE_DETECTION_DEFAULT_IMAGE,
                         'model': config.FIRE_DETECTION_MODEL
                         }

        self.image = ''
        self.model = ''

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(self.to_fetch['image'], self.name)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_url(self.to_fetch['model'], self.name)

    def start(self):
        self.retrieve_data()
        self.tflite_runtime_interpreter()

    def run(self):
        self.start()

        image = cv2.imread(self.image)
        image = cv2.resize(image, (128, 128))
        image = np.array(image, dtype=np.float32) / 255.0

        # ONNX, Caffe and TF parsers also exist.
        parser = ann.ITfLiteParser()
        network = parser.CreateNetworkFromBinaryFile(self.model)

        graph_id = 0
        input_names = parser.GetSubgraphInputTensorNames(graph_id)
        input_binding_info = parser.GetNetworkInputBindingInfo(
            graph_id, input_names[0])
        input_tensor_id = input_binding_info[0]
        input_tensor_info = input_binding_info[1]

        # Create a runtime object that will perform inference.
        options = ann.CreationOptions()
        runtime = ann.IRuntime(options)

        # Backend choices earlier in the list have higher preference.
        preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
        opt_network, messages = ann.Optimize(
            network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

        # Load the optimized network into the runtime.
        net_id, _ = runtime.LoadNetwork(opt_network)
        # Create an inputTensor for inference.
        input_tensors = ann.make_input_tensors([input_binding_info], [image])

        # Get output binding information for an output layer by using the layer
        # name.
        output_names = parser.GetSubgraphOutputTensorNames(graph_id)
        output_binding_info = parser.GetNetworkOutputBindingInfo(
            0, output_names[0])
        output_tensors = ann.make_output_tensors([output_binding_info])

        start = timer()
        runtime.EnqueueWorkload(0, input_tensors, output_tensors)
        end = timer()
        print('Elapsed time is ', (end - start) * 1000, 'ms')

        output, output_tensor_info = ann.from_output_tensor(
            output_tensors[0][1])
        print(f"Output tensor info: {output_tensor_info}")
        print(output)
        j = np.argmax(output)
        if j == 0:
            print("Non-Fire")
        else:
            print("Fire")
