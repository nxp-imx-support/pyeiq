# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os


def run():
    pipeline = "gst-launch-1.0 v4l2src ! autovideosink"
    os.system(pipeline)


def v4l2_set_pipeline(width=640, height=480, device="/dev/video0",
                 frate="30/1", leaky="leaky=downstream max-size-buffers=1",
                 sync="sync=false emit-signals=true drop=true max-buffers=1"):

    return (("v4l2src device={} ! video/x-raw,width={},height={},framerate={} " \
             "! queue {} ! videoconvert ! appsink {}").format(device, width,
                                                              height, frate,
                                                              leaky, sync))
