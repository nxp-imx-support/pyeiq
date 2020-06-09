# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import collections
import os

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
from PIL import Image

from eiq.multimedia.v4l2 import v4l2_camera_set_pipeline, v4l2_video_set_pipeline

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple(
        'BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal,
    parallel to the x or y axis.
    """
    __slots__ = ()


class VideoDevice():
    def __int__(self):
        self.name = None
        self.caps = None
        self.default_caps = None

    def get_name(self):
        return self.name

    def get_caps(self):
        return self.caps

    def get_default_caps(self):
        return self.default_caps

    def set_name(self, name):
        self.name = name

    def set_caps(self, caps):
        self.caps = caps

    def set_default_caps(self, default):
        self.default_caps = default


class Caps():
    def __init__(self):
        self.name = None
        self.format = None
        self.width = None
        self.height = None
        self.pixel_aspect_ratio = None
        self.framerate = None

    def get_name(self):
        return self.name

    def get_format(self):
        return self.format

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_pixel_aspect_ratio(self):
        return self.pixel_aspect_ratio

    def get_framerate(self):
        return self.framerate

    def set_name(self, name):
        self.name = name

    def set_format(self, form):
        self.format = form

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_pixel_aspect_ration(self, pixel_ar):
        self.pixel_aspect_ratio = pixel_ar

    def set_framerate(self, framerate):
        self.framerate = framerate


class Devices():
    def __init__(self):
        self.devices = []

    def get_video_devices(self):
        Gst.init()
        dev_monitor = Gst.DeviceMonitor()
        dev_monitor.add_filter("Video/Source")
        dev_monitor.start()

        for dev in dev_monitor.get_devices():
            video_dev = VideoDevice()
            dev_props = dev.get_properties()
            dev_caps = dev.get_caps()

            name = dev_props.get_string("device.path")
            caps = self.get_device_caps(dev_caps.normalize())
            if caps:
                default_caps = caps[0]
            else:
                default_caps = None

            video_dev.set_name(name)
            video_dev.set_caps(caps)
            video_dev.set_default_caps(default_caps)
            self.devices.append(video_dev)

        dev_monitor.stop()

    def get_device_caps(self, dev_caps):
        caps_list = []

        for i in range(dev_caps.get_size()):
            if dev_caps.get_structure(i).get_name() != "video/x-raw":
                continue

            caps = Caps()
            caps_struct = dev_caps.get_structure(i)
            caps.set_name(caps_struct.get_name())
            caps.set_format(caps_struct.get_string("format"))
            caps.set_width(caps_struct.get_int("width")[1])
            caps.set_height(caps_struct.get_int("height")[1])
            pixel_ar = ("{}/{}".format(
                        *caps_struct.get_fraction(
                        "pixel-aspect-ratio")[1:]))
            caps.set_pixel_aspect_ration(pixel_ar)
            framerate = ("{}/{}".format(
                        *caps_struct.get_fraction(
                        "framerate")[1:]))
            caps.set_framerate(framerate)
            caps_list.append(caps)

        return caps_list


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
    devices = Devices()
    devices.get_video_devices()

    if args.video_src is not None and os.path.exists(args.video_src):
        if args.video_fwk == 'opencv':
            if not args.video_src.startswith("/dev/video"):
                return cv2.VideoCapture(args.video_src)
            else:
                return cv2.VideoCapture(int(args.video_src[10]))
        elif args.video_fwk == 'v4l2':
            if not args.video_src.startswith("/dev/video"):
                pipeline = v4l2_video_set_pipeline(args.video_src)
                return cv2.VideoCapture(pipeline)

            else:
                for device in devices.devices:
                    if device.get_name() == args.video_src:
                        dev = device
                        caps = dev.get_default_caps()
                        pipeline = v4l2_camera_set_pipeline(width=caps.get_width(),
                                                height=caps.get_height(),
                                                device=dev.get_name(),
                                                frate=caps.get_framerate())
                        return cv2.VideoCapture(pipeline)
                    else:
                        print("Invalid video device. Searching for a valid one...")
        elif args.video_src == 'gstreamer':
            #set gstreamer appsink/appsrc
            print("Framework not supported.")
            return None
        else:
            print("Framework not supported.")
            return None
    else:
        print("Invalid video device. Searching for a valid one...")
        return None

def resize_image(input_details, image, use_opencv=False):
    _, height, width, _ = input_details[0]['shape']

    if use_opencv:
        image = cv2.resize(image, (width, height))
    else:
        image = Image.open(image).resize((width, height))

    return np.expand_dims(image, axis=0)
