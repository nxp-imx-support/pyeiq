# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import os
import sys
import time

import cv2 as opencv
import numpy as np
from tflite_runtime.interpreter import Interpreter

from eiq.multimedia.utils import gstreamer_configurations
import eiq.tflite.inference as inference
from eiq.tflite.ssd.config import *
from eiq.tflite.ssd.utils import *
from eiq.utils import args_parser, retrieve_from_url, retrieve_from_id


class eIQObjectDetectionSSD(object):
    def __init__(self):
        self.args = args_parser(
            camera=True, camera_inference=True, image=True,
            label=True, model=True, webcam=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(TMP_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")
        self.media_path = os.path.join(self.base_path, "media")
        self.labeled_media_path = os.path.join(self.media_path, "labeled")

        self.image = None
        self.label = None
        self.model = None
        self.video = None

        self.class_names = None
        self.colors = None

    def image_object_detection(self):
        if not os.path.exists(self.labeled_media_path):
            try:
                Path(self.labeled_media_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                sys.exit("Path.mkdir(%s) function has failed"
                    % self.labeled_media_path)

        image = opencv.imread(self.image)
        image_name = os.path.basename(self.image)
        image_data = preprocess_image_for_tflite(image)
        out_scores, out_boxes, out_classes = self.run_detection(image_data)

        result = draw_boxes(image, out_scores, out_boxes, out_classes,
                    self.class_names, self.colors)
        opencv.imwrite(os.path.join(self.labeled_media_path, image_name),
            result, [opencv.IMWRITE_JPEG_QUALITY, 90])
        
    def real_time_object_detection(self):
        self.video = gstreamer_configurations(self.args)

        while self.video.isOpened():
            start = time.time()
            ret, frame = self.video.read()

            if ret:
                image_data = preprocess_image_for_tflite(frame)
                out_scores, out_boxes, out_classes = self.run_detection(image_data)

                result = draw_boxes(frame, out_scores, out_boxes, out_classes,
                            self.class_names, self.colors)
                end = time.time()

                t = end - start
                fps  = "Fps: {:.2f}".format(1 / t)
                opencv.putText(result, fps, (10, 30),
		                    opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2, opencv.LINE_AA)

                opencv.imshow("Object detection - ssdlite_mobilenet_v2", frame)
                if opencv.waitKey(1) & 0xFF == ord('q'):
                    break

        self.video.release()
        opencv.destroyAllWindows()

    def retrieve_data(self):
        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(OBJ_DETECTION_SSD_IMAGE,
                            self.media_path)

        if self.args.label is not None and os.path.exists(self.args.label):
            self.label = self.args.label
        else:
            self.label = retrieve_from_url(OBJ_DETECTION_SSD_LABEL,
                            self.model_path)

        if self.args.model is not None and os.path.exists(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_id(OBJ_DETECTION_SSD_MODEL_ID,
                            self.model_path, OBJ_DETECTION_SSD_MODEL_NAME)
            self.model = os.path.join(self.model,
                            OBJ_DETECTION_SSD_MODEL_NAME)

    def run_detection(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        inference.inference(self.interpreter)

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = self.interpreter.get_tensor(self.output_details[3]['index'])

        boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
        out_scores, out_boxes, out_classes = non_max_suppression(scores,
                                                boxes, classes)

        return out_scores, out_boxes, out_classes

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)
        self.class_names = read_classes(self.label)
        self.colors = generate_colors(self.class_names)

    def run(self):
        self.start()

        if not self.args.camera_inference:
            self.image_object_detection()
        else:
            self.real_time_object_detection()
