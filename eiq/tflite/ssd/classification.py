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


class eIQObjectDetectionCamera(object):
    def __init__(self):
        self.args = args_parser(
            camera=True, label=True, model=True, webcam=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(TMP_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")

        self.video = None
        self.label = None
        self.model = None

        self.threshold = 0.5

    def annotate_objects(self, image, results, label, className):
        for obj in results:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * 1280)
            xmax = int(xmax * 1280)
            ymin = int(ymin * 720)
            ymax = int(ymax * 720)

            opencv.putText(image, className[int(obj['class_id']) - 1]
                           + " " + str('%.1f' % (obj['score'] * 100)) + "%",
                           (xmin, int(ymax + .05 * xmax)),
                           opencv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            opencv.rectangle(
                image, (xmin, ymax), (xmax, ymin), (0, 0, 255), thickness=2)

    def detect_objects(self, image):
        self.set_input_tensor(image)
        inference.inference(self.interpreter)

        boxes = self.get_output_tensor(0)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = []
        for i in range(count):
            if (scores[i] >= self.threshold):
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def retrieve_data(self):
        retrieve_from_url(url=OBJ_DETECTION_CAM_MODEL,
                          name=self.model_path, unzip=True)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_path,
                                      OBJ_DETECTION_CAM_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_path,
                                      OBJ_DETECTION_CAM_MODEL_NAME)

    def set_input_tensor(self, image):
        input_tensor = self.interpreter.tensor(
                            self.input_details[0]['index'])()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        return np.squeeze(
            self.interpreter.get_tensor(self.output_details[index]['index']))

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(
                                                    self.interpreter)

    def run(self):
        self.start()

        lin = open(self.label).read().strip().split("\n")
        className = [r[r.find(" ") + 1:].split(",")[0] for r in lin]

        _, height, width, _ = self.input_details[0]['shape']

        while True:
            ret, frame = self.video.read()
            resized_frame = opencv.resize(frame, (width, height))

            results = self.detect_objects(resized_frame)

            self.annotate_objects(frame, results, self.label, className)

            opencv.imshow(TITLE_OBJECT_DETECTION_CAM, frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break

        opencv.destroyAllWindows()


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
