from PIL import Image
from socket import gethostname

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GdkPixbuf

from eiq.apps.label.parser import run_label_image_no_accel, run_label_image_accel

class SwitchLabelImage(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Label Image Switch Demo")
        
        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect('destroy', self.destroy)
        
        grid = Gtk.Grid(row_spacing = 10, column_spacing = 10, border_width = 18,)
        self.add(grid)
        self.valueReturned = []
        self.labelReturned = []
        self.valueReturnedBox = []
        self.labelReturnedBox = []
        grid.set_column_homogeneous(True)
        grid.set_row_homogeneous(True)
        
        statusBox = Gtk.Box()
        statusValueBox = Gtk.Box()
        modelBox = Gtk.Box()
        modelNameBox = Gtk.Box()
        resultBox = Gtk.Box()
        percentageBox = Gtk.Box()
        inferenceBox = Gtk.Box()
        inferenceValueBox = Gtk.Box()
        imageBox = Gtk.Box()

        for i in range(5):
            self.valueReturned.append(Gtk.Entry())
            self.labelReturned.append(Gtk.Label())
            self.valueReturnedBox.append(Gtk.Box())
            self.labelReturnedBox.append(Gtk.Box())

        
        img = Image.open('/usr/bin/tensorflow-lite-2.1.0/examples/grace_hopper.bmp')
        new_img = img.resize( (507, 606) )
        new_img.save( 'test.png', 'png')


        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("test.png", 507, 606, True)
        image = Gtk.Image()
        image.set_from_pixbuf(pixbuf)

        self.set_initial_entrys()

        statusLabel = Gtk.Label()
        statusLabel.set_markup("<b>STATUS</b>")
        self.statusValueLabel = Gtk.Label("Not running")
        modelLabel = Gtk.Label()
        modelLabel.set_markup("<b>MODEL NAME</b>")
        self.modelNameLabel = Gtk.Label("")
        resultLabel = Gtk.Label()
        resultLabel.set_markup("<b>LABELS</b>")
        percentageLabel = Gtk.Label()
        percentageLabel.set_markup("<b>RESULTS (%)</b>")
        inferenceLabel = Gtk.Label()
        inferenceLabel.set_markup("<b>INFERENCE TIME</b>")
        self.inferenceValueLabel = Gtk.Label("")

        modelBox.pack_start(modelLabel, True, True, 0)
        modelNameBox.pack_start(self.modelNameLabel, True, True, 0)
        statusBox.pack_start(statusLabel, True, True, 0)
        statusValueBox.pack_start(self.statusValueLabel, True, True, 0)
        resultBox.pack_start(resultLabel, True, True, 0)
        percentageBox.pack_start(percentageLabel, True, True, 0)
        inferenceBox.pack_start(inferenceLabel, True, True, 0)
        inferenceValueBox.pack_start(self.inferenceValueLabel, True, True, 0)


        for i in range(5):
            self.labelReturnedBox[i].pack_start(self.labelReturned[i], True, True, 0)
            self.valueReturnedBox[i].pack_start(self.valueReturned[i], True, True, 0)

        imageBox.pack_start(image, True, True, 0)

        cpu_button = Gtk.Button(label="CPU")
        cpu_button.connect("clicked", self.run_inference_cpu)
        grid.attach(cpu_button, 3, 0, 1, 1)
        
        if gethostname() == "imx8mpevk":
            npu_button = Gtk.Button(label="NPU")
        else:
            npu_button = Gtk.Button(label="GPU")
        npu_button.connect("clicked", self.run_inference_npu)
        grid.attach(npu_button, 4, 0, 1, 1)

        grid.attach(modelBox, 0, 4, 2, 1)
        grid.attach(modelNameBox, 0, 5, 2, 1)
        grid.attach(inferenceBox, 0, 6, 2, 1)
        grid.attach(inferenceValueBox, 0, 7, 2, 1)
        grid.attach(resultBox, 6, 3, 1, 1)
        grid.attach(percentageBox, 7, 3, 1, 1)

        for i in range(5):
            grid.attach(self.labelReturnedBox[i], 6, (4+i), 1, 1)
            grid.attach(self.valueReturnedBox[i], 7, (4+i), 1, 1)

        grid.attach(statusBox, 3, 11, 1, 1)
        grid.attach(statusValueBox, 4, 11, 1, 1)

        grid.attach(imageBox, 2, 1, 4, 10)

        self.show_all()

    def set_initial_entrys(self):
        for i in range(5):
            self.valueReturned[i].set_editable(False)
            self.valueReturned[i].set_can_focus(False)
            self.valueReturned[i].set_text("0%")
            self.valueReturned[i].set_alignment(xalign=0)
            self.valueReturned[i].set_progress_fraction(-1)

    def set_returned_entrys(self, value):
        for i in range(5):
            self.labelReturned[i].set_text(str(value[2+i][2]))
            #TODO: check why the bar is not updating
            self.valueReturned[i].set_text(str("%.2f" % (float(value[2+i][0])*100))+"%")

    def run_inference_cpu(self, window):
        #TODO: the next two lines do not work
        self.set_initial_entrys()
        self.statusValueLabel.set_text("Running...")
        print ("Running Inference on CPU")
        x = run_label_image_no_accel()
        self.statusValueLabel.set_text("done.")
        self.modelNameLabel.set_text(x[0])
        self.inferenceValueLabel.set_text(str("%.2f" % (float(x[1])) + " ms"))
        self.set_returned_entrys(x)
        print(x)

    def run_inference_npu(self, window):
        #TODO: the next two lines do not work
        self.set_initial_entrys()
        self.statusValueLabel.set_text("Running...")
        print ("Running Inference on NPU")
        x = run_label_image_accel()
        self.statusValueLabel.set_text("done.")
        self.modelNameLabel.set_text(x[0])
        self.inferenceValueLabel.set_text(str("%.2f" % (float(x[1])) + " ms"))
        self.set_returned_entrys(x)
        print(x)

    def destroy(self, window):
        Gtk.main_quit()

def main():
    app = SwitchLabelImage()
    Gtk.main()


if __name__ == '__main__':
    main()
