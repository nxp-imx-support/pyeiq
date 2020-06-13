# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import sys
import threading

import cv2

from eiq.multimedia.utils import GstVideo, VideoConfig


def run_inference(inference_func, image, args):
    if args.video_src:
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
    else:
        try:
            frame = cv2.imread(image, cv2.IMREAD_COLOR)
            thread = threading.Thread(target=display_image,
                                      args=inference_func(frame))
            thread.daemon = True
            thread.start()
            thread.join()
        except KeyboardInterrupt:
            sys.exit("")

    cv2.destroyAllWindows()


def display_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey()
