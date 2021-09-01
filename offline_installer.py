# Copyright 2021 NXP
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
from subprocess import Popen
import shutil
import sys

from eiq.apps.config import SWITCH_IMAGE
from eiq.apps.switch_video.config import VIDEO_SWITCH_CORE
from eiq.modules.classification.config import OBJ_CLASSIFICATION
from eiq.modules.detection.config import FACE_EYES_DETECTION, OBJ_DETECTION
from eiq.utils import check_sha1

if len(sys.argv) <= 1:
    sys.exit("You need to inform the IP address of your board.")
ip = sys.argv[1]

switch_image = {'name': "eIQSwitchLabelImage",
                'sha1': SWITCH_IMAGE['sha1'],
                'src': SWITCH_IMAGE['src']['github']}

switch_video = {'name': "eIQVideoSwitchCore",
                'sha1': VIDEO_SWITCH_CORE['sha1'],
                'src': VIDEO_SWITCH_CORE['src']['github']}

obj_classification_tflite = {'name': "eIQObjectClassificationTFLite",
                             'sha1': OBJ_CLASSIFICATION['sha1'],
                             'src': OBJ_CLASSIFICATION['src']['github']}

face_and_eyes_detection = {'name': "eIQFaceAndEyesDetection",
                           'sha1': FACE_EYES_DETECTION['sha1'],
                           'src': FACE_EYES_DETECTION['src']['github']}

obj_detection = {'name': "eIQObjectDetection",
                 'sha1': OBJ_DETECTION['sha1'],
                 'src': OBJ_DETECTION['src']['github']}


apps_demos_list = [switch_image, switch_video, obj_classification_tflite,
                   face_and_eyes_detection, obj_detection]

target = "eiq.zip"
base_dir = os.path.join(os.getcwd(), "pyeiq_data", "target")

pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

for item in apps_demos_list:
    file_name = os.path.join(base_dir, (item['name'] + ".zip"))

    if not os.path.isfile(file_name) or not check_sha1(file_name, item['sha1']):
        Popen(["wget", f"{item['src']}", "-O", f"{file_name}"]).wait()
        if not check_sha1(file_name, item['sha1']):
            os.remove(file_name)
            sys.exit(f"The checksum of {file_name} failed!\nRemoving and exiting...")

Popen(["zip", "-rj", f"{target}", f"{base_dir}"]).wait()
Popen(["scp", "-r", f"{target}", f"root@{ip}:~/"]).wait()
