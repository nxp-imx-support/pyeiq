# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

def v4l2_camera_pipeline(width, height, device, frate,
                         leaky="leaky=downstream max-size-buffers=1"):

    return (("v4l2src device={} ! video/x-raw,width={},height={},framerate={} "\
             "! queue {} ! videoconvert ! appsink").format(device, width,
                                                              height, frate,
                                                              leaky))


def v4l2_video_pipeline(device, leaky="leaky=downstream max-size-buffers=1"):

    return (("filesrc location={} ! qtdemux name=d d.video_0 ! decodebin ! queue {} ! queue ! imxvideoconvert_g2d ! videoconvert ! "\
             "appsink").format(device, leaky))
