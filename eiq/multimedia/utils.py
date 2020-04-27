# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import collections

import cv2 as opencv
import numpy as np
from PIL import Image

from eiq.multimedia.v4l2 import set_pipeline

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple(
        'BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal,
    parallel to the x or y axis.
    """
    __slots__ = ()

def make_boxes(i, boxes, class_ids, scores):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=scores[i],
        bbox=BBox(xmin=np.maximum(0.0, xmin),
                        ymin=np.maximum(0.0, ymin),
                        xmax=np.minimum(1.0, xmax),
                        ymax=np.minimum(1.0, ymax)))

def gstreamer_configurations(args):
    if args.webcam >= 0:
        return opencv.VideoCapture(args.webcam)

    return opencv.VideoCapture(set_pipeline(1280, 720, device=args.camera))

def resize_image(input_details, image, use_opencv: bool = False):
    _, height, width, _ = input_details[0]['shape']

    if use_opencv:
        image = opencv.resize(image, (width, height))
    else:
        image = Image.open(image).resize((width, height))

    return np.expand_dims(image, axis=0)
