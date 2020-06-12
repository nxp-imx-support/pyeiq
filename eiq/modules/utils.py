# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import atexit
import sys

import cv2
import numpy as np
from PIL import Image

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from eiq.multimedia.utils import VideoConfig

class GstVideo:
    def __init__(self, sink, src, inference_func):
        self.sink = sink
        self.src = src
        self.inference_func = inference_func

        self.appsource = None
        self.sink_pipeline = None
        self.src_pipeline = None

        atexit.register(self.exit_handler)

    def exit_handler(self):
        self.sink_pipeline.set_state(Gst.State.NULL)
        self.src_pipeline.set_state(Gst.State.NULL)

    def run(self):
        self.sink_pipeline = Gst.parse_launch(self.sink)
        appsink = self.sink_pipeline.get_by_name('sink')
        appsink.connect("new-sample", self.on_new_frame)

        self.src_pipeline = Gst.parse_launch(self.src)
        self.appsource = self.src_pipeline.get_by_name('src')

        self.sink_pipeline.set_state(Gst.State.PLAYING)
        bus1 = self.sink_pipeline.get_bus()
        self.src_pipeline.set_state(Gst.State.PLAYING)
        bus2 = self.src_pipeline.get_bus()

        while True:
            message = bus1.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    sink_pipeline.set_state(Gst.State.NULL)
                    src_pipeline.set_state(Gst.State.NULL)
                    sys.exit("ERROR bus 1: {}: {}".format(err, debug))

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 1: {}: {}".format(err, debug))

            message = bus2.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    sink_pipeline.set_state(Gst.State.NULL)
                    src_pipeline.set_state(Gst.State.NULL)
                    sys.exit("ERROR bus 2: {}: {}".format(err, debug))

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 2: {}: {}".format(err, debug))

    def on_new_frame(self, sink):
        sample = sink.emit("pull-sample")
        caps = sample.get_caps().get_structure(0)
        resize = (caps.get_value('height'), caps.get_value('width'), 3)

        mem = sample.get_buffer()
        success, arr = mem.map(Gst.MapFlags.READ)
        img = np.ndarray(resize, buffer=arr.data, dtype=np.uint8)

        img = self.inference_func(img)[1]
        self.appsource.emit("push-buffer", Gst.Buffer.new_wrapped(img.tobytes()))
        mem.unmap(arr)

        return Gst.FlowReturn.OK

def real_time_inference(inference_func, args):
    video_config = VideoConfig(args)
    sink, src = video_config.get_config()

    if not src:
        if (not sink) or (not sink.isOpened()):
            sys.exit("Your video device could not be initialized. Exiting...")
        while sink.isOpened():
            ret, frame = sink.read()
            if ret:
                cv2.imshow(*inference_func(frame))
            else:
                print("Your video device could not capture any image.")
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        sink.release()
    else:
        gst_video = GstVideo(sink, src, inference_func)
        gst_video.run()
