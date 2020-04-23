from eiq.apps.label.parser import run_label_image_no_accel, run_label_image_accel

# include Gtk stuff here.

x = run_label_image_no_accel()
y = run_label_image_accel()

print(x)
print(y)
