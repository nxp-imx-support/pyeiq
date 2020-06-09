# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


def run():
    pipeline = "gst-launch-1.0 v4l2src ! autovideosink"
    os.system(pipeline)


def v4l2_camera_set_pipeline(width, height, device,
                 frate, leaky="leaky=downstream max-size-buffers=1",
                 sync="sync=false emit-signals=true drop=true max-buffers=1"):

    return (("v4l2src device={} ! video/x-raw,width={},height={},framerate={} " \
             "! queue {} ! videoconvert ! appsink {}").format(device, width,
                                                              height, frate,
                                                              leaky, sync))

def v4l2_video_set_pipeline(device,
                 leaky="leaky=downstream max-size-buffers=1",
                 sync="sync=false drop=True max-buffers=1 emit-signals=True max-lateness=8000000000"):

    return (("filesrc location={} ! decodebin " \
             "! queue {} ! videoconvert ! appsink {}").format(device,
                                                              leaky, sync))
