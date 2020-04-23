import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

from eiq.apps.label.parser import run_label_image_no_accel, run_label_image_accel

class SwitchLabelImage(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Label Image Switch Demo")
        
        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect('destroy', self.destroy)
        
        grid = Gtk.Grid()
        self.add(grid)
        
        cpu_button = Gtk.Button(label="CPU")
        cpu_button.connect("clicked", self.run_inference_cpu)
        grid.add(cpu_button)
        
        npu_button = Gtk.Button(label="NPU")
        npu_button.connect("clicked", self.run_inference_npu)
        grid.attach(npu_button, 1, 0, 2, 1)
        
        model_name = Gtk.Label("Test")
        grid.attach(model_name, 1, 0, 2, 1)
        #window = Gtk.Window()
        #window.set_title("Label Image - Switch CPU vs GPU/NPU")
        #window.set_default_size(1280, 720)
        #window.set_position(Gtk.WindowPosition.CENTER)
        #window.connect('destroy', self.destroy)

        #button = Gtk.Button("Hit me!")
        #button.connect("clicked", self.button_clicked)
        #window.add(button)

        self.show_all()

    def run_inference_cpu(self, window):
        print ("Running Inference on CPU")
        x = run_label_image_no_accel()
        print(x)

    def run_inference_npu(self, window):
        print ("Running Inference on CPU")
        y = run_label_image_accel()
        print(y)


    def destroy(self, window):
        Gtk.main_quit()

def main():
    app = SwitchLabelImage()
    Gtk.main()


if __name__ == '__main__':
    main()
