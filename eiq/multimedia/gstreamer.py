# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import sys

try:
    import svgwrite
except ImportError:
    None

import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, Gtk

import numpy as np


Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

GObject.threads_init()
Gst.init(None)


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal,
    parallel to the x or y axis.
    """
    __slots__ = ()


def make_boxes(i, boxes, class_ids, scores):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(id=int(class_ids[i]), score=scores[i],
                  bbox=BBox(xmin=np.maximum(0.0, xmin),
                            ymin=np.maximum(0.0, ymin),
                            xmax=np.minimum(1.0, xmax),
                            ymax=np.minimum(1.0, ymax)))


def set_appsink_pipeline(device,
                        leaky="leaky=downstream max-size-buffers=1",
                        sync="drop=True max-buffers=1 emit-signals=True max-lateness=8000000000"):

    return (("v4l2src device={} do-timestamp=True ! videoconvert ! \
                videoscale n-threads=4 method=nearest-neighbour ! \
                video/x-raw,format=RGB,width=1920,height=1080 ! \
                queue {} ! appsink name=sink {}"
            ).format(device,
                    leaky, sync))

def set_appsrc_pipeline(width=1920, height=1080):
    return (("appsrc name=src is-live=True block=True ! \
                video/x-raw,format=RGB,width=1920,height=1080, \
                framerate=20/1,interlace-mode=(string)progressive ! \
                videoconvert ! waylandsink"
            ).format(width, height))

class GstPipeline:
    def __init__(self, pipeline, user_function, src_size):
        self.user_function = user_function
        self.running = False
        self.gstbuffer = None
        self.sink_size = None
        self.src_size = src_size
        self.box = None
        self.condition = threading.Condition()

        self.pipeline = Gst.parse_launch(pipeline)
        self.overlay = self.pipeline.get_by_name('overlay')
        self.overlaysink = self.pipeline.get_by_name('overlaysink')
        appsink = self.pipeline.get_by_name('appsink')
        appsink.connect('new-sample', self.on_new_sample)

        # Set up a pipeline bus watch to catch errors.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

    def run(self):
        # Start inference worker.
        self.running = True
        worker = threading.Thread(target=self.inference_loop)
        worker.start()

        # Run pipeline.
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            Gtk.main()
        except BaseException:
            pass

        # Clean up.
        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass
        with self.condition:
            self.running = False
            self.condition.notify_all()
        worker.join()

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
            Gtk.main_quit()
        return True

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if not self.sink_size:
            s = sample.get_caps().get_structure(0)
            self.sink_size = (s.get_value('width'), s.get_value('height'))
        with self.condition:
            self.gstbuffer = sample.get_buffer()
            self.condition.notify_all()
        return Gst.FlowReturn.OK

    def get_box(self):
        if not self.box:
            glbox = self.pipeline.get_by_name('glbox')
            if glbox:
                glbox = glbox.get_by_name('filter')
            box = self.pipeline.get_by_name('box')
            assert glbox or box
            assert self.sink_size
            if glbox:
                self.box = (glbox.get_property('x'), glbox.get_property('y'),
                            glbox.get_property('width'), glbox.get_property('height'))
            else:
                self.box = (-box.get_property('left'), -box.get_property('top'),
                            self.sink_size[0] + box.get_property(
                                'left') + box.get_property('right'),
                            self.sink_size[1] + box.get_property('top') + box.get_property('bottom'))
        return self.box

    def inference_loop(self):
        while True:
            with self.condition:
                while not self.gstbuffer and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                gstbuffer = self.gstbuffer
                self.gstbuffer = None

            # Passing Gst.Buffer as input tensor avoids 2 copies of it:
            # * Python bindings copies the data when mapping gstbuffer
            # * Numpy copies the data when creating ndarray.
            # This requires a recent version of the python3-edgetpu package. If this
            # raises an exception please make sure dependencies are up to date.
            input_tensor = gstbuffer
            svg = self.user_function(
                input_tensor, self.src_size, self.get_box())
            if svg:
                if self.overlay:
                    self.overlay.set_property('data', svg)
                if self.overlaysink:
                    self.overlaysink.set_property('svg', svg)

def camera_pipeline(video_src):
    PIPELINE = 'v4l2src device=%s ! {src_caps}' % video_src
    PIPELINE += """ ! tee name=t
        t. ! {leaky_q} ! imxvideoconvert_g2d ! {scale_caps} ! videobox name=box autocrop=true
           ! {sink_caps} ! {sink_element}
        t. ! queue ! imxvideoconvert_g2d
           ! rsvgoverlay name=overlay ! waylandsink
        """
    return PIPELINE

def video_file_pipeline(video_src):
    PIPELINE = 'filesrc location=%s ! decodebin' % video_src
    PIPELINE += """ ! tee name=t
        t. ! {leaky_q} ! imxvideoconvert_g2d ! {scale_caps} ! videobox name=box autocrop=true
           ! {sink_caps} ! {sink_element}
        t. ! queue ! imxvideoconvert_g2d
           ! rsvgoverlay name=overlay ! waylandsink
    """
    return PIPELINE

def run_pipeline(user_function,
                 src_size,
                 appsink_size,
                 videosrc='/dev/video1',
                 videofile='None',
                 videofmt='raw'):
    scale = min(
        appsink_size[0] /
        src_size[0],
        appsink_size[1] /
        src_size[1])
    scale = tuple(int(x * scale) for x in src_size)
    scale_caps = 'video/x-raw,width={width},height={height}'.format(
        width=scale[0], height=scale[1])
    if videofile != None:
        PIPELINE = video_file_pipeline(videofile)

    else:
        PIPELINE = camera_pipeline(videosrc)

    SRC_CAPS = 'video/x-raw,width={width},height={height},framerate=30/1'
    SINK_ELEMENT = 'appsink name=appsink emit-signals=true max-buffers=1 drop=true'
    SINK_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    LEAKY_Q = 'queue max-size-buffers=1 leaky=downstream'

    src_caps = SRC_CAPS.format(width=src_size[0], height=src_size[1])
    sink_caps = SINK_CAPS.format(width=appsink_size[0], height=appsink_size[1])
    pipeline = PIPELINE.format(leaky_q=LEAKY_Q,
                               src_caps=src_caps, sink_caps=sink_caps,
                               sink_element=SINK_ELEMENT, scale_caps=scale_caps)

    print('Gstreamer pipeline:\n', pipeline)

    pipeline = GstPipeline(pipeline, user_function, src_size)
    pipeline.run()
