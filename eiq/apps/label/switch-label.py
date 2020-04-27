from PIL import Image

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

        valueReturnedBox_0 = Gtk.Box()
        valueReturnedBox_1 = Gtk.Box()
        valueReturnedBox_2 = Gtk.Box()
        valueReturnedBox_3 = Gtk.Box()
        valueReturnedBox_4 = Gtk.Box()

        labelReturnedBox_0 = Gtk.Box()
        labelReturnedBox_1 = Gtk.Box()
        labelReturnedBox_2 = Gtk.Box()
        labelReturnedBox_3 = Gtk.Box()
        labelReturnedBox_4 = Gtk.Box()

        self.valueReturned_0 = Gtk.Entry()
        self.valueReturned_1 = Gtk.Entry()
        self.valueReturned_2 = Gtk.Entry()
        self.valueReturned_3 = Gtk.Entry()
        self.valueReturned_4 = Gtk.Entry()

        self.labelReturned_0 = Gtk.Label()
        self.labelReturned_1 = Gtk.Label()
        self.labelReturned_2 = Gtk.Label()
        self.labelReturned_3 = Gtk.Label()
        self.labelReturned_4 = Gtk.Label()

        
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

        labelReturnedBox_0.pack_start(self.labelReturned_0, True, True, 0)
        labelReturnedBox_1.pack_start(self.labelReturned_1, True, True, 0)
        labelReturnedBox_2.pack_start(self.labelReturned_2, True, True, 0)
        labelReturnedBox_3.pack_start(self.labelReturned_3, True, True, 0)
        labelReturnedBox_4.pack_start(self.labelReturned_4, True, True, 0)

        valueReturnedBox_0.pack_start(self.valueReturned_0, True, True, 0)
        valueReturnedBox_1.pack_start(self.valueReturned_1, True, True, 0)
        valueReturnedBox_2.pack_start(self.valueReturned_2, True, True, 0)
        valueReturnedBox_3.pack_start(self.valueReturned_3, True, True, 0)
        valueReturnedBox_4.pack_start(self.valueReturned_4, True, True, 0)
        imageBox.pack_start(image, True, True, 0)

        cpu_button = Gtk.Button(label="CPU")
        cpu_button.connect("clicked", self.run_inference_cpu)
        grid.attach(cpu_button, 3, 0, 1, 1)
        
        npu_button = Gtk.Button(label="NPU")
        npu_button.connect("clicked", self.run_inference_npu)
        grid.attach(npu_button, 4, 0, 1, 1)

        grid.attach(modelBox, 0, 4, 2, 1)
        grid.attach(modelNameBox, 0, 5, 2, 1)
        grid.attach(inferenceBox, 0, 6, 2, 1)
        grid.attach(inferenceValueBox, 0, 7, 2, 1)
        grid.attach(resultBox, 6, 3, 1, 1)
        grid.attach(percentageBox, 7, 3, 1, 1)

        grid.attach(labelReturnedBox_0, 6, 4, 1, 1)
        grid.attach(labelReturnedBox_1, 6, 5, 1, 1)
        grid.attach(labelReturnedBox_2, 6, 6, 1, 1)
        grid.attach(labelReturnedBox_3, 6, 7, 1, 1)
        grid.attach(labelReturnedBox_4, 6, 8, 1, 1)

        grid.attach(valueReturnedBox_0, 7, 4, 1, 1)
        grid.attach(valueReturnedBox_1, 7, 5, 1, 1)
        grid.attach(valueReturnedBox_2, 7, 6, 1, 1)
        grid.attach(valueReturnedBox_3, 7, 7, 1, 1)
        grid.attach(valueReturnedBox_4, 7, 8, 1, 1)

        grid.attach(statusBox, 3, 11, 1, 1)
        grid.attach(statusValueBox, 4, 11, 1, 1)

        grid.attach(imageBox, 2, 1, 4, 10)

        self.show_all()

    def set_initial_entrys(self):
        self.valueReturned_0.set_editable(False)
        self.valueReturned_0.set_can_focus(False)
        self.valueReturned_0.set_text("0%")
        self.valueReturned_0.set_alignment(xalign=0)
        self.valueReturned_0.set_progress_fraction(-1)

        self.valueReturned_1.set_editable(False)
        self.valueReturned_1.set_can_focus(False)
        self.valueReturned_1.set_text("0%")
        self.valueReturned_1.set_alignment(xalign=0)
        self.valueReturned_1.set_progress_fraction(-1)

        self.valueReturned_2.set_editable(False)
        self.valueReturned_2.set_can_focus(False)
        self.valueReturned_2.set_text("0%")
        self.valueReturned_2.set_alignment(xalign=0)
        self.valueReturned_2.set_progress_fraction(-1)

        self.valueReturned_3.set_editable(False)
        self.valueReturned_3.set_can_focus(False)
        self.valueReturned_3.set_text("0%")
        self.valueReturned_3.set_alignment(xalign=0)
        self.valueReturned_3.set_progress_fraction(-1)

        self.valueReturned_4.set_editable(False)
        self.valueReturned_4.set_can_focus(False)
        self.valueReturned_4.set_text("0%")
        self.valueReturned_4.set_alignment(xalign=0)
        self.valueReturned_4.set_progress_fraction(-1)

    def set_returned_entrys(self, value):
        self.labelReturned_0.set_text(str(value[2][2]))
        self.labelReturned_1.set_text(str(value[3][2]))
        self.labelReturned_2.set_text(str(value[4][2]))
        self.labelReturned_3.set_text(str(value[5][2]))
        self.labelReturned_4.set_text(str(value[6][2]))
        #TODO: check why the bar is not updating
        self.valueReturned_0.set_text(str("%.2f" % (float(value[2][0])*100))+"%")
        self.valueReturned_1.set_text(str("%.2f" % (float(value[3][0])*100))+"%")
        self.valueReturned_2.set_text(str("%.2f" % (float(value[4][0])*100))+"%")
        self.valueReturned_3.set_text(str("%.2f" % (float(value[5][0])*100))+"%")
        self.valueReturned_4.set_text(str("%.2f" % (float(value[6][0])*100))+"%")

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
