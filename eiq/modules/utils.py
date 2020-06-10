# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import sys

import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from eiq.multimedia.utils import gstreamer_configurations

def real_time_inference(set_src_func=None, on_new_frame_func=None, inference_func=None, args=None):
    sink, src = gstreamer_configurations(args)

    if src is None:
        if (not sink) or (not sink.isOpened()):
            sys.exit("Your video device could not be initialized. Exiting...")
        while sink.isOpened():
            ret, frame = sink.read()
            if ret:
                inference_func(frame)
            else:
                print("Your video device could not capture any image.")
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        sink.release()

    else:
        print(sink)
        print(src)
        sink_pipeline = Gst.parse_launch(sink)
        appsink = sink_pipeline.get_by_name('sink')
        appsink.connect("new-sample", on_new_frame_func, appsink)

        src_pipeline = Gst.parse_launch(src)
        appsource = src_pipeline.get_by_name('src')
        set_src_func(appsource)

        sink_pipeline.set_state(Gst.State.PLAYING)
        bus1 = sink_pipeline.get_bus()
        src_pipeline.set_state(Gst.State.PLAYING)
        bus2 = src_pipeline.get_bus()

        # Main Loop
        while True:
            message = bus1.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    print("ERROR bus 1:",err,debug)
                    sink_pipeline.set_state(Gst.State.NULL)
                    src_pipeline.set_state(Gst.State.NULL)
                    quit()

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 1:",err,debug)

                if message.type == Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    print("INFO: state on bus 2 changed from ",old_state," To: ",new_state)
            message = bus2.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    print("ERROR bus 2:",err,debug)
                    sink_pipeline.set_state(Gst.State.NULL)
                    src_pipeline.set_state(Gst.State.NULL)
                    quit()

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 2:",err,debug)

                if message.type == Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    print("INFO: state on bus 2 changed from ",old_state," To: ",new_state)
