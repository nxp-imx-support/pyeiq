# Copyright 2018 The TensorFlow Authors
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import collections
from pathlib import Path
import os
import re
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2 as opencv
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from eiq.config import BASE_DIR
import eiq.engines.tflite.inference as inference
from eiq.modules.detection.config import *
from eiq.modules.detection.utils import *
from eiq.multimedia import gstreamer
from eiq.multimedia.utils import gstreamer_configurations, make_boxes
from eiq.utils import args_parser, retrieve_from_url, retrieve_from_id

try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False


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


class eIQObjectDetectionGStreamer(object):
    def __init__(self):
        self.args = args_parser(camera=True, videopath=True, webcam=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")

        self.model = None
        self.label = None

        self.videosrc = None
        self.videofile = None
        self.videofmt = "raw"
        self.src_width = 640
        self.src_height = 480

    def retrieve_data(self):
        retrieve_from_id(OBJ_DETECTION_CV_GST_MODEL_ID, self.model_path,
                         self.__class__.__name__ + ZIP, True)
        self.model = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_GST_MODEL_NAME)
        self.label = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_GST_LABEL_NAME)

    def video_src_config(self):
        if self.args.webcam >= 0:
            self.videosrc = "/dev/video" + str(self.args.webcam)
        else:
            self.videosrc = "/dev/video" + str(self.args.camera)

    def video_file_config(self):
        if self.args.videopath != 0 and os.path.exists(self.args.videopath):
            self.videofile = self.args.videopath
            self.src_width = 1920
            self.src_height = 1080

    def input_image_size(self):
        """Returns input size as (width, height, channels) tuple."""
        _, height, width, channels = self.input_details[0]['shape']
        return width, height, channels

    def input_tensor(self):
        """Returns input tensor view as numpy array of shape (height, width, channels)."""
        return self.interpreter.tensor(self.input_details[0]['index'])()[0]

    def set_input(self, buf):
        """Copies data to input tensor."""
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            np_buffer = np.reshape(np.frombuffer(
                mapinfo.data, dtype=np.uint8), self.input_image_size())
            self.input_tensor()[:, :] = np_buffer
            buf.unmap(mapinfo)

    def output_tensor(self, i):
        """Returns dequantized output tensor if quantized before."""
        output_data = np.squeeze(self.interpreter.tensor(self.output_details[i]['index'])())
        if 'quantization' not in self.output_details:
            return output_data
        scale, zero_point = self.output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def avg_fps_counter(self, window_size):
        window = collections.deque(maxlen=window_size)
        prev = time.monotonic()
        yield 0.0

        while True:
            curr = time.monotonic()
            window.append(curr - prev)
            prev = curr
            yield len(window) / sum(window)

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def shadow_text(self, dwg, x, y, text, font_size=20):
        dwg.add(dwg.text(text, insert=(x + 1, y + 1),
                         fill='black', font_size=font_size))
        dwg.add(
            dwg.text(
                text,
                insert=(
                    x,
                    y),
                fill='white',
                font_size=font_size))

    def generate_svg(self, src_size, inference_size,
                     inference_box, objs, labels, text_lines):
        dwg = svgwrite.Drawing('', size=src_size)
        src_w, src_h = src_size
        inf_w, inf_h = inference_size
        box_x, box_y, box_w, box_h = inference_box
        scale_x, scale_y = src_w / box_w, src_h / box_h

        for y, line in enumerate(text_lines, start=1):
            self.shadow_text(dwg, 10, y * 20, line)
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Absolute coordinates, input tensor space.
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            self.shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
        return dwg.tostring()

    def get_output(self, score_threshold=0.1, top_k=3, image_scale=1.0):
        """Returns list of detected objects."""
        boxes = self.output_tensor(0)
        category_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        return [make_boxes(i, boxes, category_ids, scores) for i in range(top_k) if scores[i] >= score_threshold]

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video_src_config()
        self.video_file_config()
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        if not has_svgwrite:
            sys.exit("The module svgwrite needed to run this demo was not " \
                     "found. If you want to install it type 'pip3 install " \
                     " svgwrite' at your terminal. Exiting...")

        self.start()
        labels = self.load_labels(self.label)
        w, h, _ = self.input_image_size()
        inference_size = (w, h)
        fps_counter = self.avg_fps_counter(30)

        def user_callback(input_tensor, src_size, inference_box):
            nonlocal fps_counter
            start_time = time.monotonic()
            self.set_input(input_tensor)
            inference.inference(self.interpreter)
            objs = self.get_output()
            end_time = time.monotonic()
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(fps_counter))),
            ]
            print(' '.join(text_lines))
            return self.generate_svg(
                src_size, inference_size, inference_box, objs, labels, text_lines)

        result = gstreamer.run_pipeline(user_callback,
                                        src_size=(self.src_width, self.src_height),
                                        appsink_size=inference_size,
                                        videosrc=self.videosrc,
                                        videofile=self.videofile,
                                        videofmt=self.videofmt)


class eIQObjectDetectionImage(object):
    def __init__(self):
        self.args = args_parser( label=True, model=True)
        self.labels = None
        self.model_caffe = None
        self.model_proto = None
        self.nn = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")
        self.media_path = os.path.join(self.base_path, "media")
        self.labeled_media_path = os.path.join(self.media_path, "labeled")

        self.blob = 0
        self.coordinates = []
        self.files = []
        self.hf = 0
        self.ir = 0
        self.ic = 0
        self.wf = 0

        self.ext = (".jpeg", ".jpg", ".png", ".bmp")
        self.normalize = 127.5
        self.opacity = 0.2
        self.scale_factor = 0.009718
        self.threshold = 0.2
        self.width = 300
        self.height = 300

    def caffeInference(self, img, img_name, nn):
        self.blob = opencv.dnn.blobFromImage(self.ir, self.scale_factor,
            (self.height, self.width),
            (self.normalize, self.normalize, self.normalize), False)

        nn.setInput(self.blob)
        det = nn.forward()
        self.ic = img.copy()
        cols = self.ir.shape[1]
        rows = self.ir.shape[0]

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]

            if (confidence > self.threshold):
                index = int(det[0, 0, i, 1])
                self.math(i, det, cols, rows)
                opencv.rectangle(self.ir,
                    (self.coordinates[0], self.coordinates[1]),
                    (self.coordinates[2], self.coordinates[3]),
                    (0, 255, 0))
                opencv.rectangle(self.ic,
                    (self.coordinates[4], self.coordinates[5]),
                    (self.coordinates[6], self.coordinates[7]),
                    (0, 255, 0), -1)

            self.coordinates = []

        opencv.addWeighted(self.ic, self.opacity, img,
            (1 - self.opacity), 0, img)

        labels = self.load_labels_from_file()

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]

            if (confidence > self.threshold):
                index = int(det[0, 0, i, 1])
                self.math(i, det, cols, rows)
                opencv.rectangle(img,
                    (self.coordinates[4], self.coordinates[5]),
                    (self.coordinates[6], self.coordinates[7]),
                    (0, 0, 0), 2)

                if (index in labels):
                    label = (labels[index] + ": " + str(confidence))
                    size, line = opencv.getTextSize(
                                    label, opencv.FONT_HERSHEY_TRIPLEX, 1, 3)
                    self.coordinates[5] = max(self.coordinates[5], size[1])
                    opencv.rectangle(img,
                        (self.coordinates[4], self.coordinates[5] - size[1]),
                        (self.coordinates[4] + size[0],
                        self.coordinates[5] + line),
                        (255, 255, 255), opencv.FILLED)
                    opencv.putText(img, label,
                        (self.coordinates[4], self.coordinates[5]),
                        opencv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

            self.coordinates = []

        self.save_labeled_image(img, img_name)

    def load_images_from_media_folder(self):
        for image in os.listdir(self.media_path):
            if (image.endswith(self.ext)):
                self.files.append(image)

    def load_labels_from_file(self):
        labels = {}

        with open(self.labels) as f:
            for line in f:
                (key, val) = line.split()
                labels[int(key)] = val

        return labels

    def math(self, i, d, c, r):
        self.coordinates.append(int(d[0, 0, i, 3] * c))
        self.coordinates.append(int(d[0, 0, i, 4] * r))
        self.coordinates.append(int(d[0, 0, i, 5] * c))
        self.coordinates.append(int(d[0, 0, i, 6] * r))
        self.coordinates.append(int(self.wf * int(d[0, 0, i, 3] * c)))
        self.coordinates.append(int(self.hf * int(d[0, 0, i, 4] * r)))
        self.coordinates.append(int(self.wf * int(d[0, 0, i, 5] * c)))
        self.coordinates.append(int(self.hf * int(d[0, 0, i, 6] * r)))

    def retrieve_data(self):
        retrieve_from_id(OBJ_DETECTION_IMG_MODEL_ID, self.base_path,
                         self.__class__.__name__ + ZIP, True)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.labels = self.args.label
        else:
            self.labels = os.path.join(self.model_path,
                                       OBJ_DETECTION_IMG_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model_caffe = self.args.model
        else:
            self.model_caffe =  os.path.join(self.model_path,
                                             OBJ_DETECTION_IMG_CAFFE_NAME)
            self.model_proto =  os.path.join(self.model_path,
                                             OBJ_DETECTION_IMG_PROTO_NAME)

    def save_labeled_image(self, i, n):
        if not os.path.exists(self.labeled_media_path):
            try:
                Path(self.labeled_media_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                sys.exit("Path.mkdir(%s) function has failed"
                    % self.labeled_media_path)

        opencv.imwrite(os.path.join(self.labeled_media_path, n), i)

    def start(self):
        self.retrieve_data()
        self.load_images_from_media_folder()
        self.nn = opencv.dnn.readNetFromCaffe(self.model_proto,
                                              self.model_caffe)

    def run(self):
        self.start()

        for img_name in self.files:
            img = opencv.imread(os.path.join(self.media_path, img_name))
            self.ir = opencv.resize(img, (self.height, self.width))
            self.hf = (img.shape[0] / self.height)
            self.wf = (img.shape[1] / self.width)
            self.caffeInference(img, img_name, self.nn)


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
        retrieve_from_id(OBJ_DETECTION_CV_GST_MODEL_ID, self.model_path,
                         self.__class__.__name__ + ZIP, True)
        self.model = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_GST_MODEL_NAME)
        self.label = os.path.join(self.model_path,
                                  OBJ_DETECTION_CV_GST_LABEL_NAME)

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
        retrieve_from_id(OBJ_DETECTION_SSD_MODEL_ID, self.base_path,
                         self.__class__.__name__ + ZIP, True)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_path,
                                      OBJ_DETECTION_SSD_MEDIA_NAME)

        if self.args.label is not None and os.path.exists(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_path,
                                      OBJ_DETECTION_SSD_LABEL_NAME)

        if self.args.model is not None and os.path.exists(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_path,
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
