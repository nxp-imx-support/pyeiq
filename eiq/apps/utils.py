# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import re
import subprocess

from eiq.apps.config import RUN_LABEL_IMAGE, REGEX_GET_STRING, REGEX_GET_INTEGER_FLOAT


def run(use_accel, binary, model, image, labels):
    if use_accel:
        accel = 1
    else:
        accel = 0

    run_li = RUN_LABEL_IMAGE.format(binary, model, image, labels, accel)
    return subprocess.check_output(run_li, shell=True, stderr=subprocess.STDOUT)


def get_chances(line):
    x = re.findall(REGEX_GET_INTEGER_FLOAT, line)
    y = re.sub(REGEX_GET_STRING, u'', line, flags=re.UNICODE)
    y = y.lstrip()
    x.append(y)
    return x


def parser_cpu_gpu(data, include_accel):
    parsed_data = []
    res = [string.strip() for string in data.splitlines()]

    model_name = res[0].rsplit('/', 1)[1]
    parsed_data.append(model_name)

    if include_accel:
        index = 4
        start_index = 5
    else:
        index = 3
        start_index = 4

    average_time = res[index].rsplit(':', 1)[1]
    average_time = average_time.rsplit(' ', 1)[0]
    parsed_data.append(average_time)
        
    for i in range(start_index, len(res)):
        parsed_data.append(get_chances(res[i]))
    return parsed_data


def run_label_image(accel, binary, model, image, labels):
    to_be_parsed = run(accel, binary, model, image, labels).decode('utf-8')
    return parser_cpu_gpu(to_be_parsed, accel)
