# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import re
import subprocess

from eiq.apps.label import config


def run(use_accel: bool, image_path: str):
    if image_path is None:
        image_path = config.DEFAULT_TFLITE_IMAGE

    if use_accel:                                                                          
        accel_flag = config.TFLITE_LABEL_IMAGE_ACCEL.format(image_path)           
    else:                                                                                  
        accel_flag = config.TFLITE_LABEL_IMAGE_NO_ACCEL.format(image_path)

    return subprocess.check_output(accel_flag, shell=True, stderr=subprocess.STDOUT)                                                             

def get_chances(line: str):
    x = re.findall(config.REGEX_GET_INTEGER_FLOAT, line)
    y = re.sub(config.REGEX_GET_STRING, u'', line, flags=re.UNICODE)
    y = y.lstrip()
    x.append(y)
    return x
                                                                                         
def parser_cpu_gpu(data: str, include_accel: bool):
    parsed_data = []
    l = [str.strip() for str in data.splitlines()]

    model_name = l[0].rsplit('/',1)[1]
    parsed_data.append(model_name)
    
    if include_accel:
        index = 4
        start_index = 5
    else:
        index = 3
        start_index = 4
        
    average_time = l[index].rsplit(':', 1)[1]
    average_time = average_time.rsplit(' ', 1)[0]
    parsed_data.append(average_time)
        
    for i in range(start_index, len(l)):
        parsed_data.append(get_chances(l[i]))
    return parsed_data
    
def run_label_image_no_accel(image_path: str = None):
    to_be_parsed_no_accel = run(False, image_path)
    to_be_parsed_no_accel_decoded = to_be_parsed_no_accel.decode('utf-8')
    return parser_cpu_gpu(to_be_parsed_no_accel_decoded, False)

def run_label_image_accel(image_path: str = None):
    to_be_parsed_accel = run(True, image_path)
    to_be_parsed_accel_decoded = to_be_parsed_accel.decode('utf-8')
    return parser_cpu_gpu(to_be_parsed_accel_decoded, True)


# FOR CPU

#'Loaded model /usr/bin/tensorflow-lite-2.1.0/examples/mobilenet_v1_1.0_224_quant.tflite\n
#resolved reporter\n
#invoked \n
#average time: 227.1 ms \n
#0.780392: 653 military uniform\n
#0.105882: 907 Windsor tie\n
#0.0156863: 458 bow tie\n
#0.0117647: 466 bulletproof vest\n
#0.00784314: 835 suit\n'

#return ("mobilenet_v1_1.0_224_quant.tflite", 227.1, [0.780392, 653, "military uniform"], [0.105882, 907, "Windsor tie"], [0.0156863, 458, "bow tie"], [0.0117647, 466, "bulletproof vest"], [0.00784314, 835, "suit"])


# FOR NPU/GPU

#'Loaded model /usr/bin/tensorflow-lite-2.1.0/examples/mobilenet_v1_1.0_224_quant.tflite\n
#resolved reporter\n
#INFO: Created TensorFlow Lite delegate for NNAPI.\n
#Applied NNAPI delegate.invoked \n
#average time: 3.13841 ms \n
#0.74902: 653 military uniform\n
#0.121569: 907 Windsor tie\n
#0.0196078: 458 bow tie\n
#0.0117647: 466 bulletproof vest\n
#0.00784314: 835 suit\n'

#return ("mobilenet_v1_1.0_224_quant.tflite", 3.13841, [0.74902, 653, "military uniform"], [0.121569, 907, "Windsor tie"], [0.0196078, 458, "bow tie"], [0.0117647, 466, "bulletproof vest"], [0.00784314, 835, "suit"])
