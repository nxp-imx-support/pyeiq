# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import re
import subprocess

from PIL import Image

from eiq.apps import config


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

def convert_image_to_png(image: str, width: int = 640, height: int = 480):
    image_name = "example.png"
    img_converted = Image.open(image).resize((width, height))
    img_converted.save(image_name, 'png')
    return image_name
    
