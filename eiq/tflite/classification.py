import argparse
import collections
import cv2 as opencv
import numpy as np
import os
from PIL import Image
import re
import sys
from tflite_runtime.interpreter import Interpreter
import time

from eiq.utils import retrieve_from_url, timeit, args_parser
from eiq.multimedia.v4l2 import set_pipeline
from eiq.tflite.utils import get_label, get_model

import eiq.tflite.config as config

class eIQObjectDetection(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(model = True, label = True)
        self.name = self.__class__.__name__
        self.video = ""
        self.tensor = 0
        self.to_fetch = config.OBJECT_RECOGNITION_MODEL
      
        self.label = ""
        self.model = ""
        self.pipeline = ""
        self.threshold = 0.5

    def retrieve_data(self):
        self.path = retrieve_from_url(self.to_fetch, self.name)

        if self.args.label is not None and os.path.isfile(self.args.label):
            self.label = self.args.label
        else:
            self.label = get_label(self.path)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = get_model(self.path)

    def gstreamer_configurations(self):
        self.pipeline = set_pipeline(1280, 720)
        self.video = opencv.VideoCapture(self.pipeline)

    def tflite_runtime_interpreter(self):
        self.interpreter = Interpreter(self.model)

    def set_input_tensor(self, image):
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        output_details = self.interpreter.get_output_details()[index]
        self.tensor = np.squeeze(
            self.interpreter.get_tensor(output_details['index']))

    def inference(self):
        with timeit("Inference time"):
                self.interpreter.invoke()

    def detect_objects(self, image):

        self.set_input_tensor(image)
        self.inference()

        self.get_output_tensor(0)
        boxes = self.tensor

        self.get_output_tensor(1)
        classes = self.tensor

        self.get_output_tensor(2)
        scores = self.tensor

        self.get_output_tensor(3)
        count = int(self.tensor)

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

    def annotate_objects(self, image, results, label, className):
        for obj in results:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * 1280)
            xmax = int(xmax * 1280)
            ymin = int(ymin * 720)
            ymax = int(ymax * 720)

            opencv.putText(image, className[int(obj['class_id'])-1]
                           + " " + str('%.1f'%(obj['score']*100)) + "%",
                           (xmin, int(ymax + .05 * xmax)),
                           opencv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            opencv.rectangle(
                image, (xmin, ymax), (xmax, ymin), (0, 0, 255), thickness=2)

    def start(self):
        self.retrieve_data()
        self.gstreamer_configurations()
        self.tflite_runtime_interpreter()

    def run(self):
        self.start()

        lin = open(self.label).read().strip().split("\n")
        className = [r[r.find(" ") + 1:].split(",")[0] for r in lin]

        self.interpreter.allocate_tensors()
        _, input_height, input_width, _ = self.interpreter.get_input_details(
            )[0]['shape']

        while True:
            ret, frame = self.video.read()
            rows, cols, channels = frame.shape
            resized_frame = opencv.resize(frame, (300, 300))

            results = self.detect_objects(resized_frame)

            self.annotate_objects(frame, results, self.label, className)

            opencv.imshow("eIQ PyTflite 2.1 - SSD Model", frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break
        opencv.destroyAllWindows()

class eIQLabelImage(object):
    def __init__(self, **kwargs):
        self.args = args_parser(image = True, model = True, label = True)
        self.__dict__.update(kwargs)
        self.name = self.__class__.__name__
        self.tensor = 0
        self.to_fetch = {   'image' : config.LABEL_IMAGE_DEFAULT_IMAGE,
                            'labels' : config.LABEL_IMAGE_LABELS,
                            'model' : config.LABEL_IMAGE_MODEL
        }

        self.image = ""
        self.label = ""
        self.model = ""
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
            self.label = get_label(retrieve_from_url(self.to_fetch['labels'], self.name))

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = get_model(retrieve_from_url(self.to_fetch['model'], self.name))

    def tflite_runtime_interpreter(self):
        self.interpreter = Interpreter(self.model)

    def start(self):
        self.retrieve_data()
        self.tflite_runtime_interpreter()

    def run(self):
        self.start()

        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        img = Image.open(self.image).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        with timeit("Inference time"):
            self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
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
        self.args = args_parser(image = True, model = True)
        self.name = self.__class__.__name__
        self.tensor = 0
        self.to_fetch = {   'image' : config.FIRE_DETECTION_DEFAULT_IMAGE,
                            'model' : config.FIRE_DETECTION_MODEL
        }

        self.image = ''
        self.model = ''

    def retrieve_data(self):
        if self.args.image is not None and os.path.isfile(self.args.image):
            self.image = self.args.image
        else:
            self.image = retrieve_from_url(self.to_fetch['image'], self.name)

        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_url(self.to_fetch['model'], self.name)

    def tflite_runtime_interpreter(self):
        self.interpreter = Interpreter(self.model)

    def start(self):
        self.retrieve_data()
        self.tflite_runtime_interpreter()

    def run(self):
        self.start()

        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        _,height,width,_ = input_details[0]['shape']
        floating_model = False
        if input_details[0]['dtype'] == np.float32:
            floating_model = True

        image = opencv.imread(self.image)
        image = opencv.resize(image, (width, height))
        image = np.expand_dims(image, axis=0)
        if floating_model:
            image = np.array(image, dtype=np.float32) / 255.0
        print(image.shape)

        # Test model on image.
        self.interpreter.set_tensor(input_details[0]['index'], image)

        with timeit("Inference time"):
            self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        j = np.argmax(output_data)
        if j == 0:
            print("Non-Fire")
        else:
            print("Fire")

class eIQFireDetectionCamera(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(model = True)
        self.name = self.__class__.__name__
        self.video = ""
        self.tensor = 0
        self.to_fetch = config.FIRE_DETECTION_MODEL

        self.model = ""
        self.pipeline = ""

    def retrieve_data(self):
        if self.args.model is not None and os.path.isfile(self.args.model):
            self.model = self.args.model
        else:
            self.model = retrieve_from_url(self.to_fetch, self.name)

    def gstreamer_configurations(self):
        self.pipeline = set_pipeline(1280, 720)
        self.video = opencv.VideoCapture(self.pipeline)

    def tflite_runtime_interpreter(self):
        self.interpreter = Interpreter(self.model)

    def inference(self):
        with timeit("Inference time"):
                self.interpreter.invoke()

    def detect_fire(self, image):
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        _,height,width,_ = input_details[0]['shape']
        floating_model = False
        if input_details[0]['dtype'] == np.float32:
            floating_model = True

        img = opencv.resize(image, (width, height))
        img = np.expand_dims(img, axis=0)
        if floating_model:
            img = np.array(img, dtype=np.float32) / 255.0

        # Test model on image.
        self.interpreter.set_tensor(input_details[0]['index'], img)

        self.inference()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return np.argmax(output_data)

    def start(self):
        self.retrieve_data()
        self.gstreamer_configurations()
        self.tflite_runtime_interpreter()

    def run(self):
        self.start()

        self.interpreter.allocate_tensors()

        while True:
            ret, frame = self.video.read()
            rows, cols, channels = frame.shape
            #resized_frame = opencv.resize(frame, (300, 300))

            has_fire = self.detect_fire(frame)

            if has_fire == 0:
                opencv.putText(frame, "No Fire", (200, 200), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            else:
                opencv.putText(frame, "Fire Detected", (200, 200), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            opencv.imshow("eIQ PyTflite 2.1 - " + self.name, frame)
            if (opencv.waitKey(1) & 0xFF == ord('q')):
                break
        opencv.destroyAllWindows()

class eIQCameraOpenCV(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.args = args_parser(model = True)
        self.name = self.__class__.__name__
        self.Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
        self.to_fetch = config.CAMERA_OPENCV_MODEL

        self.model = ""
        self.pipeline = ""

    def retrieve_data(self):
        path = retrieve_from_url(self.to_fetch, self.name)
        self.model = get_model(path)
        self.label = get_label(path)

    def tflite_runtime_interpreter(self):
        self.interpreter = Interpreter(self.model)

    def inference(self):
        with timeit("Inference time"):
                self.interpreter.invoke()

    def set_input(self, interpreter, image, resample=Image.NEAREST):
        """Copies data to input tensor."""
        image = image.resize((self.input_image_size(interpreter)[0:2]), resample)
        self.input_tensor(interpreter)[:, :] = image

    def input_image_size(self, interpreter):
        """Returns input image size as (width, height, channels) tuple."""
        _, height, width, channels = interpreter.get_input_details()[0]['shape']
        return width, height, channels

    def input_tensor(self, interpreter):
        """Returns input tensor view as numpy array of shape (height, width, 3)."""
        tensor_index = interpreter.get_input_details()[0]['index']
        return interpreter.tensor(tensor_index)()[0]

    def output_tensor(self, interpreter, i):
        """Returns dequantized output tensor if quantized before."""
        output_details = interpreter.get_output_details()[i]
        output_data = np.squeeze(interpreter.tensor(output_details['index'])())
        if 'quantization' not in output_details:
            return output_data
        scale, zero_point = output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def load_labels(self, path):
        p = re.compile(r'\s*(\d+)(.+)')
        with open(path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())
            return {int(num): text.strip() for num, text in lines}

    def get_output(self, interpreter, score_threshold, top_k, image_scale=1.0):
        """Returns list of detected objects."""
        boxes = self.output_tensor(interpreter, 0)
        class_ids = self.output_tensor(interpreter, 1)
        scores = self.output_tensor(interpreter, 2)
        count = int(self.output_tensor(interpreter, 3))

        def make(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return self.Object(
                id=int(class_ids[i]),
                score=scores[i],
                bbox=self.BBox(xmin=np.maximum(0.0, xmin),
                        ymin=np.maximum(0.0, ymin),
                        xmax=np.minimum(1.0, xmax),
                        ymax=np.minimum(1.0, ymax)))

        return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

    def append_objs_to_img(self, opencv_im, objs, labels):
        height, width, channels = opencv_im.shape
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

            opencv_im = opencv.rectangle(opencv_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            opencv_im = opencv.putText(opencv_im, label, (x0, y0+30),
                                opencv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return opencv_im

    class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
        """Bounding box.
        Represents a rectangle which sides are either vertical or horizontal, parallel
        to the x or y axis.
        """
        __slots__ = ()

    def start(self):
        self.retrieve_data()

    def run(self):
        self.start()

        default_model_dir = os.path.join("/tmp", "eiq", self.__class__.__name__)
        default_model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
        default_labels = 'coco_labels.txt'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help='.tflite model path',
                            default=os.path.join(default_model_dir,default_model))
        parser.add_argument('--labels', help='label file path',
                            default=os.path.join(default_model_dir, default_labels))
        parser.add_argument('--top_k', type=int, default=3,
                            help='number of categories with highest score to display')
        parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
        parser.add_argument('--threshold', type=float, default=0.1,
                            help='classifier score threshold')
        args = parser.parse_args()

        print('Loading {} with {} labels.'.format(args.model, args.labels))
        interpreter = Interpreter(args.model)
        interpreter.allocate_tensors()
        labels = self.load_labels(args.labels)

        cap = opencv.VideoCapture(args.camera_idx)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            opencv_im = frame

            opencv_im_rgb = opencv.cvtColor(opencv_im, opencv.COLOR_BGR2RGB)
            pil_im = Image.fromarray(opencv_im_rgb)

            self.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs = self.get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
            opencv_im = self.append_objs_to_img(opencv_im, objs, labels)

            opencv.imshow('frame', opencv_im)
            if opencv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        opencv.destroyAllWindows()
