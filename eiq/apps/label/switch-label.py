# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
from PIL import Image
from socket import gethostname
import threading

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf, GLib

from eiq.apps.label.parser import run_label_image_no_accel, run_label_image_accel
from eiq.utils import args_parser, retrieve_from_id

from eiq.apps.label import config

class SwitchLabelImage(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Label Image Switch Demo")
        self.args = args_parser(image=True)
        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)

        grid = Gtk.Grid(row_spacing = 10, column_spacing = 10, border_width = 18,)
        self.add(grid)
        self.hw_accel = self.get_hw_accel()
        self.valueReturned = []
        self.labelReturned = []
        self.valueReturnedBox = []
        self.labelReturnedBox = []
        self.displayedImage = Gtk.Image()
        self.image = config.DEFAULT_TFLITE_IMAGE
        self.imageMap = Gtk.ListStore(str)

        self.images_path = retrieve_from_id(config.IMAGES_DRIVE_ID, "switch-images", config.IMAGES_DRIVE_NAME + ".zip",unzip_flag=True)
        self.images_path = os.path.join(self.images_path, config.IMAGES_DRIVE_NAME)

        self.get_bmp_images()

        grid.set_column_homogeneous(True)
        grid.set_row_homogeneous(True)

        modelBox = Gtk.Box()
        modelNameBox = Gtk.Box()
        resultBox = Gtk.Box()
        percentageBox = Gtk.Box()
        inferenceBox = Gtk.Box()
        inferenceValueBox = Gtk.Box()
        imageLabelBox = Gtk.Box()
        imageMapBox = Gtk.Box()
        imageBox = Gtk.Box()

        self.imageComboBox = Gtk.ComboBox.new_with_model(self.imageMap)
        self.imageComboBox.connect("changed", self.on_combo_image_changed)
        imageRenderedList = Gtk.CellRendererText()
        self.imageComboBox.pack_start(imageRenderedList, True)
        self.imageComboBox.add_attribute(imageRenderedList, "text", 0)

        for i in range(5):
            self.valueReturned.append(Gtk.Entry())
            self.labelReturned.append(Gtk.Label())
            self.valueReturnedBox.append(Gtk.Box())
            self.labelReturnedBox.append(Gtk.Box())

        if self.args.image is not None and os.path.exists(self.args.image):
            img = Image.open(self.args.image)
        else:
            img = Image.open(self.image)

        self.set_displayed_image(self.image)
        self.set_initial_entrys()

        modelLabel = Gtk.Label()
        modelLabel.set_markup("<b>MODEL NAME</b>")
        self.modelNameLabel = Gtk.Label.new(None)
        resultLabel = Gtk.Label()
        resultLabel.set_markup("<b>LABELS</b>")
        percentageLabel = Gtk.Label()
        percentageLabel.set_markup("<b>RESULTS (%)</b>")
        inferenceLabel = Gtk.Label()
        inferenceLabel.set_markup("<b>INFERENCE TIME</b>")
        self.inferenceValueLabel = Gtk.Label.new(None)
        imageLabel = Gtk.Label()
        imageLabel.set_markup("<b>SELECT AN IMAGE</b>")
        imageLabel.set_xalign(0.0)

        modelBox.pack_start(modelLabel, True, True, 0)
        modelNameBox.pack_start(self.modelNameLabel, True, True, 0)
        resultBox.pack_start(resultLabel, True, True, 0)
        percentageBox.pack_start(percentageLabel, True, True, 0)
        inferenceBox.pack_start(inferenceLabel, True, True, 0)
        inferenceValueBox.pack_start(self.inferenceValueLabel, True, True, 0)
        imageLabelBox.pack_start(imageLabel, True, True, 0)
        imageMapBox.pack_start(self.imageComboBox, True, True, 0)

        for i in range(5):
            self.labelReturnedBox[i].pack_start(self.labelReturned[i], True, True, 0)
            self.valueReturnedBox[i].pack_start(self.valueReturned[i], True, True, 0)

        imageBox.pack_start(self.displayedImage, True, True, 0)

        self.cpu_button = Gtk.Button.new_with_label("CPU")
        self.cpu_button.connect("clicked", self.run_cpu_inference)
        grid.attach(self.cpu_button, 3, 0, 1, 1)
        self.npu_button = Gtk.Button.new_with_label(self.hw_accel)
        self.npu_button.connect("clicked", self.run_npu_inference)
        grid.attach(self.npu_button, 4, 0, 1, 1)

        grid.attach(modelBox, 0, 5, 2, 1)
        grid.attach(modelNameBox, 0, 6, 2, 1)
        grid.attach(inferenceBox, 0, 7, 2, 1)
        grid.attach(inferenceValueBox, 0, 8, 2, 1)
        grid.attach(resultBox, 6, 3, 1, 1)
        grid.attach(percentageBox, 7, 3, 1, 1)

        for i in range(5):
            grid.attach(self.labelReturnedBox[i], 6, (4+i), 1, 1)
            grid.attach(self.valueReturnedBox[i], 7, (4+i), 1, 1)

        grid.attach(imageLabelBox, 0, 2, 2, 1)
        grid.attach(imageMapBox, 0, 3, 2, 1)
        grid.attach(imageBox, 2, 1, 4, 10)

    def set_displayed_image(self, image):
        new_img = Image.open(image).resize( (507, 606) )
        new_img.save( 'test.png', 'png')

        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("test.png", 507, 606, True)
        self.displayedImage.set_from_pixbuf(pixbuf)

    def on_combo_image_changed(self, combo):
        iterr = combo.get_active_iter()
        if iterr is not None:
            model = combo.get_model()
            imageName = model[iterr][0]
            print("Selected image: %s" % imageName)
            self.image = os.path.join(self.images_path, imageName)
            self.set_displayed_image(os.path.join(self.images_path, imageName))

    def get_bmp_images(self):
        for file in os.listdir(self.images_path):
            self.imageMap.append([file])

    def get_hw_accel(self):
        if gethostname() == "imx8mpevk":
            return "NPU"

        return "GPU"

    def set_initial_entrys(self):
        for i in range(5):
            self.labelReturned[i].set_text("")
            self.valueReturned[i].set_editable(False)
            self.valueReturned[i].set_can_focus(False)
            self.valueReturned[i].set_text("0%")
            self.valueReturned[i].set_alignment(xalign=0)
            self.valueReturned[i].set_progress_fraction(-1)

    def set_returned_entrys(self, value):
        x = 0
        for i in value[2:]:
            self.labelReturned[x].set_text(str(i[2]))
            self.valueReturned[x].set_progress_fraction(float(i[0]))
            self.valueReturned[x].set_text(str("%.2f" % (float(i[0])*100))+"%")
            x = x + 1

    def set_pre_inference(self):
        self.set_initial_entrys()
        self.modelNameLabel.set_text("")
        self.inferenceValueLabel.set_text("Running...")
        self.imageComboBox.set_sensitive(False)
        self.cpu_button.set_sensitive(False)
        self.npu_button.set_sensitive(False)

    def set_post_inference(self, x):
        self.modelNameLabel.set_text(x[0])
        self.inferenceValueLabel.set_text(str("%.2f" % (float(x[1])) + " ms"))
        self.set_returned_entrys(x)
        self.imageComboBox.set_sensitive(True)
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
    app = SwitchLabelImage()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
