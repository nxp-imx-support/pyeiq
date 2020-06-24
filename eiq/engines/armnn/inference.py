# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import cv2 as opencv
import numpy as np
import pyarmnn as ann
from pyarmnn import __version__ as pyarmnn_version
from eiq.utils import Timer


class aNNInterpreter:
    def __init__(self, model=None):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.inference_time = None
        # ONNX, Caffe and TF parsers also exist.
        if model is not None:
            parser = ann.ITfLiteParser()
            network = parser.CreateNetworkFromBinaryFile(model)

            graph_id = 0
            input_names = parser.GetSubgraphInputTensorNames(graph_id)
            self.input_binding_info = parser.GetNetworkInputBindingInfo(
                graph_id, input_names[0])
            input_tensor_id = self.input_binding_info[0]
            input_tensor_info = self.input_binding_info[1]
            print(input_tensor_info.GetShape())

            # Create a runtime object that will perform inference.
            options = ann.CreationOptions()
            self.runtime = ann.IRuntime(options)

            # Backend choices earlier in the list have higher preference.
            preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
            opt_network, messages = ann.Optimize(
                network, preferredBackends, self.runtime.GetDeviceSpec(), ann.OptimizerOptions())

            # Load the optimized network into the runtime.
            net_id, _ = self.runtime.LoadNetwork(opt_network)

            output_names = parser.GetSubgraphOutputTensorNames(graph_id)
            self.output_binding_info = parser.GetNetworkOutputBindingInfo(
                0, output_names[0])
            self.output_tensors = ann.make_output_tensors([self.output_binding_info])
        
    def set_tensor(self, image):
        self.input_tensors = ann.make_input_tensors([self.input_binding_info], [image])

    def get_tensor(self, index, squeeze=False):
        if squeeze:
            return np.squeeze(ann.workload_tensors_to_ndarray(self.output_tensors))

        return ann.workload_tensors_to_ndarray(self.output_tensors)

    def run_inference(self):
        timer = Timer()
        with timer.timeit("Inference time armNN"):
            self.runtime.EnqueueWorkload(0, self.input_tensors, self.output_tensors)
        self.inference_time = timer.time
