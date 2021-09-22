# Copyright 2019 Google LLC
# Copyright 2021 NXP
#
# This file was copied from Google LLC respecting its rights. All the modified
# parts below are according to Google LLC's LICENSE terms.
#
# SPDX-License-Identifier:    Apache-2.0

import collections
import sys

try:
    import svgwrite
except ImportError:
    None

import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, Gtk

import numpy as np


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

GObject.threads_init()
Gst.init(None)


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    __slots__ = ()


def set_appsink_video_pipeline(device,width,height,
                        leaky="leaky=2 max-size-buffers=2"):

    return (("filesrc location={} ! qtdemux name=d d.video_0 ! \
                queue ! decodebin ! videorate ! videoconvert ! \
                videoscale n-threads=4 method=nearest-neighbour ! \
                video/x-raw,format=RGB,width={},height={} ! \
                queue {} ! appsink name=sink emit-signals=True"
            ).format(device,width,height,
                    leaky))

def set_appsink_pipeline(device,width,height,
                        leaky="leaky=2 max-size-buffers=2"):

    return (("v4l2src device={} do-timestamp=True ! videoconvert ! \
                videoscale n-threads=4 method=nearest-neighbour ! \
                video/x-raw,format=RGB,width={},height={} ! \
                queue {} ! appsink name=sink emit-signals=True"
            ).format(device,width,height,
                    leaky))

def set_appsrc_pipeline(width, height):
    return (("appsrc name=src is-live=True block=True ! \
                video/x-raw,format=RGB,width={},height={}, \
                framerate=30/1,interlace-mode=(string)progressive ! \
                videoconvert ! imxvideoconvert_g2d ! waylandsink"
            ).format(width, height))
