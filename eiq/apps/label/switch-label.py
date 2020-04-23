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
        
        statusBox = Gtk.Box()
        statusValueBox = Gtk.Box()
        modelBox = Gtk.Box()
        modelNameBox = Gtk.Box()
        resultBox = Gtk.Box()
        percentageBox = Gtk.Box()
        inferenceBox = Gtk.Box()
        inferenceValueBox = Gtk.Box()

        statusLabel = Gtk.Label("Status: ")
        self.statusValueLabel = Gtk.Label("Not running")
        modelLabel = Gtk.Label("Model: ")
        self.modelNameLabel = Gtk.Label("")
        resultLabel = Gtk.Label("Results: ")
        percentageLabel = Gtk.Label("%")
        inferenceLabel = Gtk.Label("Inference: ")
        self.inferenceValueLabel = Gtk.Label("")

        modelBox.pack_start(modelLabel, True, True, 0)
        modelNameBox.pack_start(self.modelNameLabel, True, True, 0)
        statusBox.pack_start(statusLabel, True, True, 0)
        statusValueBox.pack_start(self.statusValueLabel, True, True, 0)
        resultBox.pack_start(resultLabel, True, True, 0)
        percentageBox.pack_start(percentageLabel, True, True, 0)
        inferenceBox.pack_start(inferenceLabel, True, True, 0)
        inferenceValueBox.pack_start(self.inferenceValueLabel, True, True, 0)

        cpu_button = Gtk.Button(label="CPU")
        cpu_button.connect("clicked", self.run_inference_cpu)
        grid.attach(cpu_button, 1, 0, 1, 1)
        
        npu_button = Gtk.Button(label="NPU")
        npu_button.connect("clicked", self.run_inference_npu)
        grid.attach(npu_button, 2, 0, 1, 1)

        grid.attach(modelBox, 1, 1, 1, 1)
        grid.attach(modelNameBox, 2, 1, 3, 1)
        grid.attach(statusBox, 5, 5, 1, 1)
        grid.attach(statusValueBox, 6, 5, 1, 1)
        grid.attach(resultBox, 1, 3, 1, 1)
        grid.attach(percentageBox, 2, 3, 1, 1)
        grid.attach(inferenceBox, 1, 2, 1, 1)
        grid.attach(inferenceValueBox, 2, 2, 2, 1)

        self.show_all()

    def run_inference_cpu(self, window):
        self.statusValueLabel.set_text("Running...")
        print ("Running Inference on CPU")
        x = run_label_image_no_accel()
        self.statusValueLabel.set_text("Finished.")
        self.modelNameLabel.set_text(x[0])
        self.inferenceValueLabel.set_text(x[1])
        print(x)

    def run_inference_npu(self, window):
        self.statusValueLabel.set_text("Running...")
        print ("Running Inference on NPU")
        x = run_label_image_accel()
        self.statusValueLabel.set_text("Finished.")
        self.modelNameLabel.set_text(x[0])
        self.inferenceValueLabel.set_text(x[1])
        print(x)

    def destroy(self, window):
        Gtk.main_quit()

def main():
    app = SwitchLabelImage()
    Gtk.main()


if __name__ == '__main__':
    main()
