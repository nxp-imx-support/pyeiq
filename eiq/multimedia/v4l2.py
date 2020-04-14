import os


def run():
    pipeline = "gst-launch-1.0 v4l2src ! autovideosink"
    os.system(pipeline)
    

def set_pipeline(width: int, height: int, device: str = "/dev/video0",
                 frate: int = 30,
                 leaky: str = "leaky=downstream max-size-buffers=1",
                 sync: str = "sync=false emit-signals=true drop=true max-buffers=1"):

    return (("""v4l2src device={} ! video/x-raw,width={},height={},framerate={}/1! queue {} ! videoconvert ! appsink {}""").format(device, width, height, frate, leaky, sync))
