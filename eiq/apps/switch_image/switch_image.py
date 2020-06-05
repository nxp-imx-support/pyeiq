# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
from PIL import Image
from socket import gethostname
import threading

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf, GLib

from eiq.apps import config
from eiq.apps.utils import convert_image_to_png
from eiq.apps.utils import run_label_image_no_accel, run_label_image_accel
from eiq.config import BASE_DIR
from eiq.utils import args_parser, Downloader


class eIQSwitchLabelImage(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title=config.TITLE_LABEL_IMAGE_SWITCH)
        self.args = args_parser(download=True, image=True)
        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
        self.media_dir = os.path.join(self.base_dir, "media")

        grid = Gtk.Grid(row_spacing = 10, column_spacing = 10, border_width = 18,)
        self.add(grid)
        self.hw_accel = self.get_hw_accel()
        self.value_returned = []
        self.label_returned = []
        self.value_returned_box = []
        self.label_returned_box = []
        self.displayed_image = Gtk.Image()
        self.image = config.DEFAULT_TFLITE_IMAGE
        self.image_map = Gtk.ListStore(str)

        download = Downloader(self.args)
        download.retrieve_data(config.SWITCH_IMAGES_MEDIA_SRC,
                               self.__class__.__name__ + config.ZIP,
                               self.base_dir, config.SWITCH_IMAGES_MEDIA_SHA1,
                               True)

        self.get_bmp_images()

        grid.set_column_homogeneous(True)
        grid.set_row_homogeneous(True)

        model_box = Gtk.Box()
        model_name_box = Gtk.Box()
        result_box = Gtk.Box()
        percentage_box = Gtk.Box()
        inference_box = Gtk.Box()
        inference_value_box = Gtk.Box()
        image_label_box = Gtk.Box()
        image_map_box = Gtk.Box()
        image_box = Gtk.Box()

        self.image_combo_box = Gtk.ComboBox.new_with_model(self.image_map)
        self.image_combo_box.connect("changed", self.on_combo_image_changed)
        image_rendered_list = Gtk.CellRendererText()
        self.image_combo_box.pack_start(image_rendered_list, True)
        self.image_combo_box.add_attribute(image_rendered_list, "text", 0)

        for i in range(5):
            self.value_returned.append(Gtk.Entry())
            self.label_returned.append(Gtk.Label())
            self.value_returned_box.append(Gtk.Box())
            self.label_returned_box.append(Gtk.Box())

        if self.args.image is not None and os.path.exists(self.args.image):
            if str(self.args.image).endswith(".bmp"):
                self.image = self.args.image

        self.set_displayed_image(self.image)
        self.set_initial_entrys()

        model_label = Gtk.Label()
        model_label.set_markup(config.SWITCH_MODEL_NAME)
        self.model_name_label  = Gtk.Label.new(None)
        result_label = Gtk.Label()
        result_label.set_markup(config.SWITCH_LABELS)
        percentage_label = Gtk.Label()
        percentage_label.set_markup(config.SWITCH_RESULTS)
        inference_label = Gtk.Label()
        inference_label.set_markup(config.SWITCH_INFERENCE_TIME)
        self.inference_value_label = Gtk.Label.new(None)
        image_label = Gtk.Label()
        image_label.set_markup(config.SWITCH_SELECT_IMAGE)
        image_label.set_xalign(0.0)

        model_box.pack_start(model_label, True, True, 0)
        model_name_box.pack_start(self.model_name_label , True, True, 0)
        result_box.pack_start(result_label, True, True, 0)
        percentage_box.pack_start(percentage_label, True, True, 0)
        inference_box.pack_start(inference_label, True, True, 0)
        inference_value_box.pack_start(self.inference_value_label, True, True, 0)
        image_label_box.pack_start(image_label, True, True, 0)
        image_map_box.pack_start(self.image_combo_box, True, True, 0)

        for i in range(5):
            self.label_returned_box[i].pack_start(self.label_returned[i], True, True, 0)
            self.value_returned_box[i].pack_start(self.value_returned[i], True, True, 0)

        image_box.pack_start(self.displayed_image, True, True, 0)

        self.cpu_button = Gtk.Button.new_with_label("CPU")
        self.cpu_button.connect("clicked", self.run_cpu_inference)
        grid.attach(self.cpu_button, 3, 0, 1, 1)
        self.npu_button = Gtk.Button.new_with_label(self.hw_accel)
        self.npu_button.connect("clicked", self.run_npu_inference)
        grid.attach(self.npu_button, 4, 0, 1, 1)

        grid.attach(model_box, 0, 5, 2, 1)
        grid.attach(model_name_box, 0, 6, 2, 1)
        grid.attach(inference_box, 0, 7, 2, 1)
        grid.attach(inference_value_box, 0, 8, 2, 1)
        grid.attach(result_box, 6, 3, 1, 1)
        grid.attach(percentage_box, 7, 3, 1, 1)

        for i in range(5):
            grid.attach(self.label_returned_box[i], 6, (4+i), 1, 1)
            grid.attach(self.value_returned_box[i], 7, (4+i), 1, 1)

        grid.attach(image_label_box, 0, 2, 2, 1)
        grid.attach(image_map_box, 0, 3, 2, 1)
        grid.attach(image_box, 2, 1, 4, 10)

    def set_displayed_image(self, image):
        image_converted = convert_image_to_png(image)
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_converted, 507, 606, True)
        self.displayed_image.set_from_pixbuf(pixbuf)

    def on_combo_image_changed(self, combo):
        iterr = combo.get_active_iter()
        if iterr is not None:
            model = combo.get_model()
            image_name = model[iterr][0]
            print("Selected image: {}".format(image_name))
            self.image = os.path.join(self.media_dir, image_name)
            self.set_displayed_image(self.image)

    def get_bmp_images(self):
        for file in os.listdir(self.media_dir):
            self.image_map.append([file])

    def get_hw_accel(self):
        if gethostname() == "imx8mpevk":
            return "NPU"

        return "GPU"

    def set_initial_entrys(self):
        for i in range(5):
            self.label_returned[i].set_text("")
            self.value_returned[i].set_editable(False)
            self.value_returned[i].set_can_focus(False)
            self.value_returned[i].set_text("0%")
            self.value_returned[i].set_alignment(xalign=0)
            self.value_returned[i].set_progress_fraction(-1)

    def set_returned_entrys(self, value):
        x = 0
        for i in value[2:]:
            self.label_returned[x].set_text(str(i[2]))
            self.value_returned[x].set_progress_fraction(float(i[0]))
            self.value_returned[x].set_text("{0:.2f}%".format(float(i[0])*100))
            x = x + 1

    def set_pre_inference(self):
        self.set_initial_entrys()
        self.model_name_label .set_text("")
        self.inference_value_label.set_text("Running...")
        self.image_combo_box.set_sensitive(False)
        self.cpu_button.set_sensitive(False)
        self.npu_button.set_sensitive(False)

    def set_post_inference(self, x):
        self.model_name_label.set_text(x[0])
        self.inference_value_label.set_text("{0:.2f} ms".format(float(x[1])))
        self.set_returned_entrys(x)
        self.image_combo_box.set_sensitive(True)
        self.cpu_button.set_sensitive(True)
        self.npu_button.set_sensitive(True)

        for i in x:
            print(i)

    def run_cpu_inference(self, window):
        self.set_pre_inference()
        thread = threading.Thread(target=self.run_inference, args=(False,))
        thread.daemon = True
        thread.start()

    def run_inference(self, accel):
        if accel:
            print ("Running Inference on {0}".format(self.hw_accel))
            x = run_label_image_accel(self.image)
        else:
            print ("Running Inference on CPU")
            x = run_label_image_no_accel(self.image)

        GLib.idle_add(self.set_post_inference, x)

    def run_npu_inference(self, window):
        self.set_pre_inference()
        thread = threading.Thread(target=self.run_inference, args=(True,))
        thread.daemon = True
        thread.start()

def main():
    app = eIQSwitchLabelImage()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
