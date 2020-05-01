import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk

import config


class MLPlayer(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title=config.MAIN_WINDOW_TITLE)
        self.set_default_size(640, 480)
        self.set_position(Gtk.WindowPosition.CENTER)


def main():
    app = MLPlayer()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()