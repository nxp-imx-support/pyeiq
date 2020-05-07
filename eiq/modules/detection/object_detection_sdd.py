# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import os
import re
import sys
import time

import cv2 as opencv
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from eiq.config import BASE_DIR
import eiq.engines.tflite.inference as inference
from eiq.modules.detection.config import *
from eiq.modules.detection.utils import *
from eiq.multimedia.utils import gstreamer_configurations, make_boxes
from eiq.utils import args_parser, retrieve_from_url, retrieve_from_id


class eIQObjectDetectionCamera(object):
    def __init__(self):
        self.args = args_parser(
            camera=True, label=True, model=True, webcam=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
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


class eIQObjectDetectionOpenCV(object):
    def __init__(self):
        self.args = args_parser(camera=True, webcam=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")

        self.video = None
        self.model = None
        self.label = None

    def retrieve_data(self):
        retrieve_from_url(url=OBJ_DETECTION_CV_MODEL,
                          name=self.model_path, unzip=True)
        self.model = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_MODEL_NAME)
        self.label = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_LABEL_NAME)

    def set_input(self, image, resample=Image.NEAREST):
        """Copies data to input tensor."""
        image = image.resize(
            (self.input_image_size()[0:2]), resample)
        self.input_tensor()[:, :] = image

    def input_image_size(self):
        """Returns input image size as (width, height, channels) tuple."""
        _, height, width, channels = self.input_details[0]['shape']
        return width, height, channels

    def input_tensor(self):
        """Returns input tensor view as numpy array of shape (height, width, 3)."""
        return self.interpreter.tensor(self.input_details[0]['index'])()[0]

    def output_tensor(self, i):
        """Returns dequantized output tensor if quantized before."""
        output_data = np.squeeze(self.interpreter.tensor(
                                    self.output_details[i]['index'])())
        if 'quantization' not in self.output_details:
            return output_data
        scale, zero_point = self.output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def get_output(self, score_threshold=0.1, top_k=3, image_scale=1.0):
        """Returns list of detected objects."""
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)
        count = int(self.output_tensor(3))

        return [make_boxes(
                    i, boxes, class_ids, scores) for i in range(
                    top_k) if scores[i] >= score_threshold]

    def append_objs_to_img(self, opencv_im, objs, labels):
        height, width, channels = opencv_im.shape
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x0, y0, x1, y1 = int(
                x0 * width), int(
                y0 * height), int(x1 * width), int(y1 * height)

            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            opencv_im = opencv.rectangle(
                opencv_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            opencv_im = opencv.putText(opencv_im, label, (x0, y0 + 30),
                                       opencv.FONT_HERSHEY_SIMPLEX, 1.0,
                                       (255, 0, 0), 2)
        return opencv_im

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(
                                                    self.interpreter)

    def run(self):
        self.start()
        labels = self.load_labels(self.label)

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            opencv_im = frame

            opencv_im_rgb = opencv.cvtColor(opencv_im, opencv.COLOR_BGR2RGB)
            pil_im = Image.fromarray(opencv_im_rgb)

            self.set_input(pil_im)
            inference.inference(self.interpreter)
            objs = self.get_output()
            opencv_im = self.append_objs_to_img(opencv_im, objs, labels)

            opencv.imshow(TITLE_OBJECT_DETECTION_CV, opencv_im)
            if opencv.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        opencv.destroyAllWindows()


class eIQObjectDetectionSSD(object):
    def __init__(self):
        self.args = args_parser(
            camera=True, camera_inference=True, image=True,
            label=True, model=True, webcam=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_path = os.path.join(self.base_path, "media")
        self.model_path = os.path.join(self.base_path, "model")
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

                opencv.imshow(TITLE_OBJECT_DETECTION_SSD, frame)
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
