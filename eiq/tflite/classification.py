# Copyright 2018 The TensorFlow Authors
#
## Copyright 2020 NXP Semiconductors
##
## This file was copied from TensorFlow respecting its rights. All the modified
## parts below are according to TensorFlow's LICENSE terms.
##
## SPDX-License-Identifier:    Apache-2.0

import argparse
import collections

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import os
import re
import sys
import time

import cv2 as opencv
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from eiq.multimedia import gstreamer
from eiq.multimedia.utils import gstreamer_configurations, make_boxes, resize_image
from eiq.multimedia.v4l2 import set_pipeline
from eiq.utils import retrieve_from_url, retrieve_from_id, timeit, args_parser

import eiq.tflite.config as config
import eiq.tflite.inference as inference
from eiq.tflite.utils import get_label, get_model_from_path, get_model_from_zip


try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False


class eIQObjectDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(
            camera=True, webcam=True, model=True, label=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = config.OBJECT_RECOGNITION_MODEL

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
        self.path = retrieve_from_url(self.to_fetch, self.name)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = get_label(self.path)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = get_model_from_zip(self.path)

    def set_input_tensor(self, image):
        input_tensor = self.interpreter.tensor(self.input_details[0]['index'])()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        return np.squeeze(
            self.interpreter.get_tensor(self.output_details[index]['index']))

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

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

            opencv.imshow(config.TITLE_OBJECT_DETECTION_CAMERA, frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break
        opencv.destroyAllWindows()


class eIQLabelImage(object):
    def __init__(self, **kwargs):
        self.args = args_parser(image=True, model=True, label=True)
        self.__dict__.update(kwargs)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = {'image': config.LABEL_IMAGE_DEFAULT_IMAGE,
                         'labels': config.LABEL_IMAGE_LABELS,
                         'model': config.LABEL_IMAGE_MODEL
                         }

        self.image = None
        self.label = None
        self.model = None

        self.input_mean = 127.5
        self.input_std = 127.5

    def load_labels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(self.to_fetch['image'], self.name)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = retrieve_from_url(self.to_fetch['labels'], self.name)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_url(self.to_fetch['model'], self.name)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        image = resize_image(self.input_details, self.image, use_opencv=False)
        floating_model = self.input_details[0]['dtype'] == np.float32

        if floating_model:
            image = (np.float32(image) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(self.label)

        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


class eIQFireDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(image=True, model=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = {'image': config.FIRE_DETECTION_DEFAULT_IMAGE,
                         'model': config.FIRE_DETECTION_MODEL,
                         }

        self.image = None
        self.model = None

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(self.to_fetch['image'], self.name)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = get_model_from_path(retrieve_from_id(self.to_fetch['model'], self.name, self.name + ".tflite"))

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        image = opencv.imread(self.image)
        image = resize_image(self.input_details, image, use_opencv=True)

        if self.input_details[0]['dtype'] == np.float32:
            image = np.array(image, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if np.argmax(output_data) == 0:
            print("Non-Fire")
        else:
            print("Fire")


class eIQFireDetectionCamera(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(camera=True, webcam=True, model=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = config.FIRE_DETECTION_MODEL

        self.model = None
        self.video = None

        self.msg = "No Fire"
        self.msg_color = (0, 255, 0)

    def retrieve_data(self):
        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = get_model_from_path(retrieve_from_id(self.to_fetch, self.name, self.name + ".tflite"))

    def detect_fire(self, image):
        img = resize_image(self.input_details, image, use_opencv=True)
        if self.input_details[0]['dtype'] == np.float32:
            img = np.array(img, dtype=np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        inference.inference(self.interpreter)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output_data)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

    def run(self):
        self.start()

        while True:
            ret, frame = self.video.read()
            if self.detect_fire(frame) == 0:
                self.message = "No Fire"
                self.color = (0, 255, 0)
            else:
                self.message = "Fire Detected!"
                self.color = (0, 0, 255)

            opencv.putText(frame, self.message, (50, 50),
                            opencv.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
            opencv.imshow(config.TITLE_FIRE_DETECTION_CAMERA, frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break
        opencv.destroyAllWindows()


class eIQObjectDetectionOpenCV(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(camera=True, webcam=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = config.CAMERA_OPENCV_MODEL

        self.video = None
        self.model = None
        self.label = None

    def retrieve_data(self):
        path = os.path.dirname(get_model_from_zip(retrieve_from_url(self.to_fetch, self.name)))
        self.model = os.path.join(path, config.CAMERA_OPENCV_DEFAULT_MODEL)
        self.label = os.path.join(path, config.CAMERA_OPENCV_DEFAULT_LABEL)

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
        output_data = np.squeeze(self.interpreter.tensor(self.output_details[i]['index'])())
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

        return [make_boxes(i, boxes, class_ids, scores) for i in range(top_k) if scores[i] >= score_threshold]

    def append_objs_to_img(self, opencv_im, objs, labels):
        height, width, channels = opencv_im.shape
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x0, y0, x1, y1 = int(
                x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            opencv_im = opencv.rectangle(
                opencv_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            opencv_im = opencv.putText(opencv_im, label, (x0, y0 + 30),
                                       opencv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return opencv_im

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video = gstreamer_configurations(self.args)
        self.retrieve_data()
        self.interpreter = inference.load_model(self.model)
        self.input_details, self.output_details = inference.get_details(self.interpreter)

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

            opencv.imshow(config.TITLE_OBJECT_DETECTION_OPENCV, opencv_im)
            if opencv.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        opencv.destroyAllWindows()


class eIQObjectDetectionGStreamer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(camera=True, webcam=True)
        self.name = self.__class__.__name__
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.to_fetch = config.CAMERA_GSTREAMER_MODEL

        self.model = None
        self.label = None

        self.videosrc = None
        self.videofmt = "raw"

    def retrieve_data(self):
        path = os.path.dirname(
            get_model_from_zip(retrieve_from_url(self.to_fetch, self.name)))
        self.model = os.path.join(path, config.CAMERA_GSTREAMER_DEFAULT_MODEL)
        self.label = os.path.join(path, config.CAMERA_GSTREAMER_DEFAULT_LABEL)

    def video_src_config(self):
        if self.args.webcam >= 0:
            self.videosrc = "/dev/video" + str(self.args.webcam)
        else:
            self.videosrc = "/dev/video" + str(self.args.camera)

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
                                        src_size=(640, 480),
                                        appsink_size=inference_size,
                                        videosrc=self.videosrc,
                                        videofmt=self.videofmt)
