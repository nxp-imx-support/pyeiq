# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import sys

import cv2

from eiq.multimedia.utils import gstreamer_configurations

def real_time_inference(inference_func, args):
    video = gstreamer_configurations(args)
    if (not video) or (not video.isOpened()):
        sys.exit("Your video device could not be initialized. Exiting...")

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            inference_func(frame)
        else:
            print("Your video device could not capture any image.")
            break
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    video.release()
