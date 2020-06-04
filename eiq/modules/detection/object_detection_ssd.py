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

from eiq.config import BASE_DIR
import eiq.engines.tflite.inference as inference
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.helper.overlay import OpenCVOverlay
from eiq.modules.detection.config import *
from eiq.modules.detection.utils import *
from eiq.multimedia import gstreamer
from eiq.multimedia.utils import gstreamer_configurations, make_boxes
from eiq.utils import args_parser, Downloader

try:
    import svgwrite
    has_svgwrite = True
except ImportError:
    has_svgwrite = False


class eIQObjectsDetection:
    def __init__(self):
        self.args = args_parser(download=True, image=True,label=True,
                                model=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.interpreter = None
        self.image = None
        self.label = None
        self.model = None

        self.class_names = None
        self.class_names_dict = {}
        self.colors = None

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_DETECTION_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               OBJ_DETECTION_MODEL_SHA1, True)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      OBJ_DETECTION_MEDIA_NAME)

        if self.args.label is not None and os.path.exists(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_dir,
                                      OBJ_DETECTION_LABEL_NAME)

        if self.args.model is not None and os.path.exists(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_dir,
                                      OBJ_DETECTION_MODEL_NAME)
        
    def dictionary(self):
        with open(self.label) as f:
            i = 0
            for line in f:
                _id = line.split()
                self.class_names_dict[np.float32(_id[0])] = i
                i = i + 1

    def load_labels(self, label_path):
        with open(label_path) as f:
            labels = {}
            for line in f.readlines():
                m = re.match(r"(\d+)\s+(\w+)", line.strip())
                labels[int(m.group(1))] = m.group(2)
            return labels

    def process_image(self, image):
        self.interpreter.set_tensor(np.expand_dims(image, axis=0))
        self.interpreter.run_inference()
        
        positions = self.interpreter.get_tensor(0, squeeze=True)
        classes = self.interpreter.get_tensor(1, squeeze=True)
        scores = self.interpreter.get_tensor(2, squeeze=True)

        result = []
        for idx, score in enumerate(scores):
            if score > 0.5:
                result.append({'pos': positions[idx], '_id': classes[idx]})
        return result

    def display_result(self, result, frame, labels):
        width = frame.shape[1]
        height = frame.shape[0]

        for obj in result:
            pos = obj['pos']
            _id = obj['_id']

            x1 = int(pos[1] * width)
            x2 = int(pos[3] * width)
            y1 = int(pos[0] * height)
            y2 = int(pos[2] * height)

            top = max(0, np.floor(y1 + 0.5).astype('int32'))
            left = max(0, np.floor(x1 + 0.5).astype('int32'))
            bottom = min(height, np.floor(y2 + 0.5).astype('int32'))
            right = min(width, np.floor(x2 + 0.5).astype('int32'))

            cv2.rectangle(frame, (left, top), (right, bottom),
                          self.colors[self.class_names_dict[_id]], 6)

            label_size = cv2.getTextSize(labels[_id], FONT, FONT_SIZE,
                                         FONT_THICKNESS)[0]
            label_rect_left = int(left - 3)
            label_rect_top = int(top - 3)
            label_rect_right = int(left + 3 + label_size[0])
            label_rect_bottom = int(top - 5 - label_size[1])

            cv2.rectangle(frame, (label_rect_left, label_rect_top),
                          (label_rect_right, label_rect_bottom),
                          self.colors[self.class_names_dict[_id]], -1)
            cv2.putText(frame, labels[_id], (left, int(top - 4)),
                        FONT, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

        inference_time_overlay = OpenCVOverlay(frame, self.interpreter.inference_time)
        frame = inference_time_overlay.draw_inference_time()
        cv2.imshow('Object Detection', frame)

    def detect_objects(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((self.interpreter.width(), self.interpreter.height()))
        top_result = self.process_image(image)
        self.display_result(top_result, frame, self.label)

    def real_time_detection(self):
        video = gstreamer_configurations(self.args)
        if (not video) or (not video.isOpened()):
            sys.exit("Your video device could not be found. Exiting...")

        while True:
            ret, frame = video.read()
            if ret:
                self.detect_objects(frame)
            else:
                print("Your video device could not capture any image.\n"\
                      "Please, check your device's configurations." )
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        video.release()

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)
        self.class_names = read_classes(self.label)
        self.colors = generate_colors(self.class_names)
        self.dictionary()
        self.label = self.load_labels(self.label)

    def run(self):
        self.start()

        if self.args.video_src:
            self.real_time_detection()
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_objects(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()


class eIQObjectDetectionGStreamer:
    def __init__(self):
        self.args = args_parser(download=True, video_src=True)
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

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_DETECTION_CV_GST_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_path,
                               OBJ_DETECTION_CV_GST_MODEL_SHA1, True)

        self.model = os.path.join(self.base_path,
                                  OBJ_DETECTION_CV_GST_MODEL_NAME)
        self.label = os.path.join(self.base_path,
                                  OBJ_DETECTION_CV_GST_LABEL_NAME)

    def video_config(self):
        if self.args.video_src and self.args.video_src.startswith("/dev/video"):
            self.videosrc = self.args.video_src
        elif self.args.video_src and os.path.exists(self.args.video_src):
            self.videofile = self.args.video_src
            self.src_width = 1920
            self.src_height = 1080

    def input_image_size(self):
        _, height, width, channels = self.input_details[0]['shape']
        return width, height, channels

    def input_tensor(self):
        return self.interpreter.tensor(self.input_details[0]['index'])()[0]

    def set_input(self, buf):
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if result:
            np_buffer = np.reshape(np.frombuffer(
                mapinfo.data, dtype=np.uint8), self.input_image_size())
            self.input_tensor()[:, :] = np_buffer
            buf.unmap(mapinfo)

    def output_tensor(self, i):
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
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            x, y, w, h = int(x * inf_w), int(y *
                                             inf_h), int(w * inf_w), int(h * inf_h)
            x, y = x - box_x, y - box_y
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            self.shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
        return dwg.tostring()

    def get_output(self, score_threshold=0.1, top_k=3, image_scale=1.0):
        boxes = self.output_tensor(0)
        category_ids = self.output_tensor(1)
        scores = self.output_tensor(2)
        return [make_boxes(i, boxes, category_ids, scores) for i in range(top_k) if scores[i] >= score_threshold]

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video_config()
        self.gather_data()
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


class eIQObjectDetectionDNN:
    def __init__(self):
        self.args = args_parser(download=True, image=True, label=True,
                                model=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_dir = os.path.join(self.base_dir, "model")
        self.media_dir = os.path.join(self.base_dir, "media")

        self.image = None
        self.labels = None
        self.model_caffe = None
        self.model_proto = None
        self.nn = None

        self.normalize = 127.5
        self.scale_factor = 0.009718
        self.threshold = 0.2
        self.width = 300
        self.height = 300

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_DETECTION_DNN_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               OBJ_DETECTION_DNN_MODEL_SHA1, True)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      OBJ_DETECTION_DNN_MEDIA_NAME)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.labels = self.args.label
        else:
            self.labels = os.path.join(self.model_dir,
                                       OBJ_DETECTION_DNN_LABEL_NAME)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model_caffe = self.args.model
        else:
            self.model_caffe =  os.path.join(self.model_dir,
                                             OBJ_DETECTION_DNN_CAFFE_NAME)
            self.model_proto =  os.path.join(self.model_dir,
                                             OBJ_DETECTION_DNN_PROTO_NAME)

    def load_labels(self, labels_path):
        labels = {}

        with open(labels_path) as f:
            for line in f:
                (key, val) = line.split()
                labels[int(key)] = val

        return labels

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (self.height, self.width))
        blob = cv2.dnn.blobFromImage(image, self.scale_factor,
                                     (self.height, self.width),
                                     (self.normalize, self.normalize,
                                      self.normalize), False)
        self.nn.setInput(blob)
        det = self.nn.forward()

        for i in range(det.shape[2]):
            confidence = det[0, 0, i, 2]

            if (confidence > self.threshold):
                index = int(det[0, 0, i, 1])
                left = int(width * det[0, 0, i, 3])
                top = int(height * det[0, 0, i, 4])
                right = int(width * det[0, 0, i, 5])
                bottom = int(height * det[0, 0, i, 6])
                cv2.rectangle(frame, (left, top), (right, bottom),
                              FONT_COLOR, FONT_THICKNESS)

                if (index in self.labels):
                    label = ("{0}: {1:.3f}".format(self.labels[index],
                                                   confidence))
                    label_size = cv2.getTextSize(label, FONT, FONT_SIZE,
                                                 FONT_THICKNESS + 1)[0]
                    top = max(top, label_size[1])
                    cv2.rectangle(frame, (left, top - label_size[1]),
                                  (left + label_size[0], top),
                                  FONT_COLOR, cv2.FILLED)
                    cv2.putText(frame, label, (left, top), FONT,
                                FONT_SIZE, (255, 255, 255), FONT_THICKNESS - 1)

        cv2.imshow(TITLE_OBJECT_DETECTION_DNN, frame)

    def real_time_detection(self):
        video = gstreamer_configurations(self.args)
        if (not video) or (not video.isOpened()):
            sys.exit("Your video device could not be found. Exiting...")

        while True:
            ret, frame = video.read()
            if ret:
                self.detect_objects(frame)
            else:
                print("Your video device could not capture any image.\n"\
                      "Please, check your device's configurations.")
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        video.release()

    def start(self):
        self.gather_data()
        self.labels = self.load_labels(self.labels)
        self.nn = opencv.dnn.readNetFromCaffe(self.model_proto,
                                              self.model_caffe)

    def run(self):
        self.start()

        if self.args.video_src:
            self.real_time_detection()
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_objects(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()


class eIQObjectDetectionOpenCV:
    def __init__(self):
        self.args = args_parser(download=True, video_src=True)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.base_path = os.path.join(BASE_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")

        self.video = None
        self.model = None
        self.label = None

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_DETECTION_CV_GST_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_path,
                               OBJ_DETECTION_CV_GST_MODEL_SHA1, True)

        self.model = os.path.join(self.base_path,
                                  OBJ_DETECTION_CV_GST_MODEL_NAME)
        self.label = os.path.join(self.base_path,
                                  OBJ_DETECTION_CV_GST_LABEL_NAME)

    def set_input(self, image, resample=Image.NEAREST):
        image = image.resize(
            (self.input_image_size()[0:2]), resample)
        self.input_tensor()[:, :] = image

    def input_image_size(self):
        _, height, width, channels = self.input_details[0]['shape']
        return width, height, channels

    def input_tensor(self):
        return self.interpreter.tensor(self.input_details[0]['index'])()[0]

    def output_tensor(self, i):
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
        if (not self.video) or (not self.video.isOpened()):
            sys.exit("Your video device could not be found. Exiting...")
        self.gather_data()
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


class eIQObjectDetectionSSD:
    def __init__(self):
        self.args = args_parser(download=True, image=True, label=True,
                                model=True, video_src=True)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")
        self.model_dir = os.path.join(self.base_dir, "model")

        self.interpreter = None
        self.image = None
        self.label = None
        self.model = None

        self.class_names = None
        self.colors = None

    def detect_objects(self, frame):
        image = preprocess_image_for_tflite(frame)
        out_scores, out_boxes, out_classes = self.run_detection(image)

        result = draw_boxes(frame, out_scores, out_boxes, out_classes,
                            self.class_names, self.colors)

        cv2.imshow(TITLE_OBJECT_DETECTION_SSD, result)

    def real_time_detection(self):
        video = gstreamer_configurations(self.args)
        if (not video) or (not video.isOpened()):
            sys.exit("Your video device could not be initialized. Exiting...")

        while video.isOpened():
            ret, frame = video.read()
            if ret:
                self.detect_objects(frame)
            else:
                print("Your video device could not capture any image.\n"\
                      "Please, check your device's configurations.")
                break
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        video.release()

    def gather_data(self):
        download = Downloader(self.args)
        download.retrieve_data(OBJ_DETECTION_SSD_MODEL_SRC,
                               self.__class__.__name__ + ZIP, self.base_dir,
                               OBJ_DETECTION_SSD_MODEL_SHA1, True)

        if self.args.image is not None and os.path.exists(self.args.image):
            self.image = self.args.image
        else:
            self.image = os.path.join(self.media_dir,
                                      OBJ_DETECTION_SSD_MEDIA_NAME)

        if self.args.label is not None and os.path.exists(self.args.label):
            self.label = self.args.label
        else:
            self.label = os.path.join(self.model_dir,
                                      OBJ_DETECTION_SSD_LABEL_NAME)

        if self.args.model is not None and os.path.exists(self.args.model):
            self.model = self.args.model
        else:
            self.model = os.path.join(self.model_dir,
                                      OBJ_DETECTION_SSD_MODEL_NAME)

    def run_detection(self, image):
        self.interpreter.set_tensor(image)
        self.interpreter.run_inference()

        boxes = self.interpreter.get_tensor(0, squeeze=True)
        scores = self.interpreter.get_tensor(2, squeeze=True)
        classes = self.interpreter.get_tensor(1)
        classes = np.squeeze(classes + 1).astype(np.int32)

        return non_max_suppression(scores, boxes, classes)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.gather_data()
        self.interpreter = TFLiteInterpreter(self.model)
        self.class_names = read_classes(self.label)
        self.colors = generate_colors(self.class_names)

    def run(self):
        self.start()

        if self.args.video_src:
            self.real_time_detection()
        else:
            frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
            self.detect_objects(frame)
            cv2.waitKey()

        cv2.destroyAllWindows()
