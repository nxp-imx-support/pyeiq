import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import os
import threading
from time import time

import cv2 as opencv
from dlr import DLRModel
import numpy as np

from eiq.utils import args_parser, retrieve_from_id

from .config import *


class eIQObjectDetectionDLR(object):
    def __init__(self):
        self.args = args_parser(
            camera=True, label=True, model=True, webcam=True)

        self.base_path = os.path.join(TMP_DIR, self.__class__.__name__)
        self.model_path = os.path.join(self.base_path, "model")

        self.model = None
        self.videosrc = None
        self.appsource = None
        self.pipeline0 = None
        self.pipeline1 = None

    def video_src_config(self):
        if self.args.webcam >= 0:
            self.videosrc = "/dev/video" + str(self.args.webcam)
        else:
            self.videosrc = "/dev/video" + str(self.args.camera)

    def inference(self, img):
        #prepare image to input.Resize adding borders and normalize.
        nn_input=opencv.resize(img, (NN_IN_SIZE,
                                     int(NN_IN_SIZE / 4 * 3)))
        nn_input=opencv.copyMakeBorder(nn_input, int(NN_IN_SIZE / 8),
                                       int(NN_IN_SIZE / 8),
                                       0, 0, opencv.BORDER_CONSTANT,
                                       value=(0, 0, 0))
        nn_input=nn_input.astype('float64')
        nn_input=nn_input.reshape((NN_IN_SIZE * NN_IN_SIZE ,3))
        nn_input=np.transpose(nn_input)

        for x in range(3):
            nn_input[x,:] = nn_input[x,:] - MEAN[x]
            nn_input[x,:] = nn_input[x,:] / STD[x]

        #Run the model
        tbefore = time()
        outputs = self.model.run({'data': nn_input})
        tafter = time()
        last_inference_time = tafter - tbefore
        objects = outputs[0][0]
        scores = outputs[1][0]
        bounding_boxes = outputs[2][0]

        #Draw bounding boxes
        i = 0
        while (scores[i]>0.5):
            y1 = int(
                 (bounding_boxes[i][1] - NN_IN_SIZE / 8) * WIDTH / NN_IN_SIZE)
            x1 = int(bounding_boxes[i][0] * HEIGHT / (NN_IN_SIZE * 3 / 4))
            y2 = int(
                 (bounding_boxes[i][3] - NN_IN_SIZE / 8) * WIDTH / NN_IN_SIZE)
            x2 = int(bounding_boxes[i][2] * HEIGHT / (NN_IN_SIZE * 3 / 4))

            object_id = int(objects[i])
            opencv.rectangle(img, (x2, y2), (x1, y1),
                             COLORS[object_id % len(COLORS)], 2)
            opencv.rectangle(img, (x1 + 70, y2 + 15), (x1, y2),
                             COLORS[object_id%len(COLORS)],opencv.FILLED)
            opencv.putText(img, CLASS_NAMES[object_id], (x1, y2 + 10),
                           opencv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                           1, opencv.LINE_AA)

            i = i + 1

        opencv.rectangle(img, (110, 17), (0, 0), (0, 0, 0), opencv.FILLED)
        opencv.putText(img, "inf. time: %.3fs" % last_inference_time,
                       (3, 12), opencv.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1, opencv.LINE_AA)

    # Pipeline 1 output
    def on_new_frame(self, sink, data):

        sample = sink.emit("pull-sample")
        captured_gst_buf = sample.get_buffer()
        caps = sample.get_caps()
        im_height_in = caps.get_structure(0).get_value('height')
        im_width_in = caps.get_structure(0).get_value('width')
        mem = captured_gst_buf.get_all_memory()
        success, arr = mem.map(Gst.MapFlags.READ)
        img = np.ndarray((im_height_in,im_width_in,3),buffer=arr.data,dtype=np.uint8)
        self.inference(img)
        self.appsource.emit("push-buffer", Gst.Buffer.new_wrapped(img.tobytes()))
        mem.unmap(arr)
        return Gst.FlowReturn.OK

    def main(self):
        # SagemakerNeo init
        self.model = DLRModel(self.model_path, 'cpu')

        # Gstreamer Init
        Gst.init(None)

        pipeline1_cmd="v4l2src device="+self.videosrc+" do-timestamp=True ! videoconvert ! \
            videoscale n-threads=4 method=nearest-neighbour ! \
            video/x-raw,format=RGB,width="+str(WIDTH)+",height="+str(HEIGHT)+" ! \
            queue leaky=downstream max-size-buffers=1 ! appsink name=sink \
            drop=True max-buffers=1 emit-signals=True max-lateness=8000000000"

        pipeline2_cmd = "appsrc name=appsource1 is-live=True block=True ! \
            video/x-raw,format=RGB,width="+str(WIDTH)+",height="+ \
            str(HEIGHT)+",framerate=20/1,interlace-mode=(string)progressive ! \
            videoconvert ! waylandsink" #v4l2sink max-lateness=8000000000 device=/dev/video14"

        self.pipeline1 = Gst.parse_launch(pipeline1_cmd)
        appsink = self.pipeline1.get_by_name('sink')
        appsink.connect("new-sample", self.on_new_frame, appsink)

        self.pipeline2 = Gst.parse_launch(pipeline2_cmd)
        self.appsource = self.pipeline2.get_by_name('appsource1')

        self.pipeline1.set_state(Gst.State.PLAYING)
        bus1 = self.pipeline1.get_bus()
        self.pipeline2.set_state(Gst.State.PLAYING)
        bus2 = self.pipeline2.get_bus()

        # Main Loop
        while True:
            message = bus1.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    print("ERROR bus 1:",err,debug)
                    self.pipeline1.set_state(Gst.State.NULL)
                    self.pipeline2.set_state(Gst.State.NULL)
                    quit()

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 1:",err,debug)

                if message.type == Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    print("INFO: state on bus 2 changed from ",old_state," To: ",new_state)
            message = bus2.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err,debug = message.parse_error()
                    print("ERROR bus 2:",err,debug)
                    self.pipeline1.set_state(Gst.State.NULL)
                    self.pipeline2.set_state(Gst.State.NULL)
                    quit()

                if message.type == Gst.MessageType.WARNING:
                    err,debug = message.parse_warning()
                    print("WARNING bus 2:",err,debug)

                if message.type == Gst.MessageType.STATE_CHANGED:
                    old_state, new_state, pending_state = message.parse_state_changed()
                    print("INFO: state on bus 2 changed from ",old_state," To: ",new_state)

    def retrieve_data(self):
        retrieve_from_id(OBJ_DETECTION_DLR_MODEL_ID, self.base_path,
                         OBJ_DETECTION_DLR_MODEL_NAME, unzip_flag=True)

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        self.video_src_config()
        self.retrieve_data()

    def run(self):
        self.start()

        thread = threading.Thread(target = self.main)
        thread.start()
