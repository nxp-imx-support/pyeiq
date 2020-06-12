# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import collections
import os
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
from PIL import Image

from eiq.multimedia.gstreamer import set_appsink_pipeline, set_appsrc_pipeline
from eiq.multimedia.v4l2 import v4l2_camera_pipeline, v4l2_video_pipeline


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
            default_caps = None if not caps else caps[0]

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
            pixel_ar = ("{}/{}".format(*caps_struct.get_fraction(
                                       "pixel-aspect-ratio")[1:]))
            caps.set_pixel_aspect_ration(pixel_ar)
            framerate = ("{}/{}".format(*caps_struct.get_fraction(
                                        "framerate")[1:]))
            caps.set_framerate(framerate)
            caps_list.append(caps)

        return caps_list

    def search_device(self, dev_name):
        dev = None

        for device in self.devices:
            if device.get_name() == dev_name:
                    dev = device

            if not dev:
                print("The specified video_src does not exists.\n"
                      "Searching for default video device...")
                if self.devices:
                    dev = self.devices[0]

                if not dev:
                    sys.exit("No video device found. Exiting...")
                else:
                    print("Using {} as video device".format(dev.get_name()))

        return dev

class VideoConfig:
    def __init__(self, args):
        self.video_fwk = args.video_fwk
        self.video_src = args.video_src
        self.devices = Devices()
        self.devices.get_video_devices()

    def get_config(self):
        if self.video_fwk == "gstreamer":
            return self.gstreamer_config()
        elif self.video_fwk == "opencv":
            return self.opencv_config()
        else:
            if self.video_fwk != "v4l2":
                print("Invalid video framework. Using v4l2 instead.")

            return self.v4l2_config()

    def gstreamer_config(self):
        if self.video_src and os.path.isfile(self.video_src):
            sys.exit("Video file not supported by GStreamer framework.")
        else:
            dev = self.devices.search_device(self.video_src)
            caps = dev.get_default_caps()
            sink_pipeline = set_appsink_pipeline(device=dev.get_name())
            src_pipeline = set_appsrc_pipeline(width=caps.get_width(),
                                               height=caps.get_height())
            return sink_pipeline, src_pipeline

    def opencv_config(self):
        if self.video_src and os.path.exists(self.video_src):
            return cv2.VideoCapture(self.video_src), None
        else:
            dev = self.devices.search_device(self.video_src)
            dev = int(dev.get_name()[10:])
            return cv2.VideoCapture(dev), None

    def v4l2_config(self):
        if self.video_src and os.path.exists(self.video_src):
            pipeline = v4l2_video_pipeline(self.video_src)
        else:
            dev = self.devices.search_device(self.video_src)
            caps = dev.get_default_caps()
            pipeline = v4l2_camera_pipeline(width=caps.get_width(),
                                            height=caps.get_height(),
                                            device=dev.get_name(),
                                            frate=caps.get_framerate())

        return cv2.VideoCapture(pipeline), None
