# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys
import threading

import cv2

from eiq.config import BASE_DIR, ZIP
from eiq.multimedia.overlay import OpenCVOverlay
from eiq.multimedia.utils import GstVideo, VideoConfig
from eiq.utils import args_parser, Downloader


class DemoBase:
    def __init__(self, download=False, image=False, labels=False,
                 model=False, video_fwk=False, video_src=False,
                 class_name=None, data=None):
        self.args = args_parser(download, image, labels, model,
                                video_fwk, video_src)
        self.overlay = OpenCVOverlay()
        self.class_name = class_name

        self.base_dir = os.path.join(BASE_DIR, self.class_name)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.data = data
        self.image = None
        self.labels = None
        self.model = None

        self.interpreter = None

    def gather_data(self):
        if self.data and 'src' in self.data:
            downloader = Downloader(self.args)
            downloader.retrieve_data(self.data['src'], self.class_name + ZIP,
                                     self.base_dir, self.data['sha1'], True)

        if hasattr(self.args, 'image'):
            if self.args.image and os.path.isfile(self.args.image):
                self.image = self.args.image
        if not self.image and self.data and 'image' in self.data:
            self.image = os.path.join(self.media_dir, self.data['image'])

        if hasattr(self.args, 'labels'):
            if self.args.labels and os.path.isfile(self.args.labels):
                self.labels = self.args.labels
        if not self.labels and self.data and 'labels' in self.data:
            self.labels = os.path.join(self.model_dir, self.data['labels'])

        if hasattr(self.args, 'model'):
            if self.args.model and os.path.isfile(self.args.model):
                self.model = self.args.model
        if not self.model and self.data and 'model' in self.data:
            self.model = os.path.join(self.model_dir, self.data['model'])

    @staticmethod
    def load_labels(path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def run_inference(self, inference_func):
        if self.args.video_src:
            video_config = VideoConfig(self.args)
            sink, src = video_config.get_config()

            if not src:
                if (not sink) or (not sink.isOpened()):
                    sys.exit("Your video device could not be initialized. Exiting...")
                while sink.isOpened():
                    ret, frame = sink.read()
                    if ret:
                        cv2.imshow(self.data['window_title'], inference_func(frame))
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
                frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
                thread = threading.Thread(target=display_image,
                                          args=(self.data['window_title'],
                                                inference_func(frame)))
                thread.daemon = True
                thread.start()
                thread.join()
            except KeyboardInterrupt:
                sys.exit("")

        cv2.destroyAllWindows()


def display_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey()
