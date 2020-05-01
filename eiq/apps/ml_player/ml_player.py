import os

import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk

import config


class MLPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title=config.MAIN_WINDOW_TITLE)
        self.set_default_size(640, 480)
        self.set_position(Gtk.WindowPosition.CENTER)

        self.demo_to_run = None
        self.demos_list = self.get_demos()
        self.description = Gtk.Label.new(config.DEFAULT_DEMOS_DESCRIPTION)

        self.grid = Gtk.Grid(
            row_spacing = 10, column_spacing = 10,
            border_width = 18, expand=True
        )
        self.add(self.grid)

        self.add_demo_box(0,0,1,1)


    def add_demo_box(self, col=0, row=0, width=1, height=1):
        demos_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=10, expand=True
        )

        demos_label = Gtk.Label.new(None)
        demos_label.set_markup("<b>Select a demo</b>")
        demos_label.set_xalign(config.ALIGN_LEFT)
        demos_box.pack_start(demos_label, False, False, True)

        # Create ComboBox to select demo
        demos_combo = Gtk.ComboBoxText()
        demos_combo.set_entry_text_column(0)
        demos_combo.connect("changed", self.on_demos_combo_changed)
        for demo in self.demos_list:
            demos_combo.append_text(demo)
        demos_box.pack_start(demos_combo, False, False, True)

        demos_description_frame = Gtk.Frame.new("Demo Description")
        demos_description_frame.set_label_align(
            config.ALIGN_CENTER, config.ALIGN_CENTER
        )
        demos_description_frame.add(self.description)
        self.description.set_xalign(0.05)
        demos_box.pack_start(demos_description_frame, False, False, True)

        self.grid.attach(demos_box, col, row, width, height)

    def get_demos(self):
        demos_list = []

        if not os.path.isdir(config.DEFAULT_DEMOS_DIR):
            demos_list.append("No PyeIQ demo found")
        else:
            for demo in os.listdir(config.DEFAULT_DEMOS_DIR):
                if "image" in demo:
                    demos_list.append(demo)

        return demos_list

    def on_demos_combo_changed(self, combo):
        demo = combo.get_active_text()

        if demo is not None:
            self.demo_to_run = demo

def main():
    app = MLPlayer()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()