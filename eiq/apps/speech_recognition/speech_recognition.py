# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import cv2
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf

from config import WAV_FILES
from pulse import generate_pulse_plot


class eIQSpeechRecognition():

    def __init__(self):
        window = Gtk.Window()
        window.set_title("PyeIQ - Speech Recognition")
        window.set_default_size(1280, 720)
        window.set_position(Gtk.WindowPosition.CENTER)
        window.connect('destroy', self.destroy)

        box = Gtk.Box()
        box.set_spacing(5)
        box.set_orientation(Gtk.Orientation.VERTICAL)
        window.add(box)

        self.image = Gtk.Image()
        box.add(self.image)

        button = Gtk.Button("RIGHT")
        button.connect("clicked", self.on_open_clicked)
        box.add(button)

        window.show_all()

    def on_open_clicked(self, button):
        pulse_plot = GdkPixbuf.Pixbuf.new_from_file("pulse.png")
        self.image.set_from_pixbuf(pulse_plot)


    def destroy(self, window):
        Gtk.main_quit()

def crop_pulse_png():
    image = cv2.imread("pulse.png")
    y = 200
    x = 25
    h = -1
    w = -1
    crop_image = image[x:w, y:h]
    cv2.imwrite("pulse.png", crop_image)


def main():
    wav_file = WAV_FILES["right"]
    generate_pulse_plot(wav_file)     
    crop_pulse_png()    
    app = eIQSpeechRecognition()
    Gtk.main()

if __name__ == '__main__':
    main()
