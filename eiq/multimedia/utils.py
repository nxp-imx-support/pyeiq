import cv2 as opencv
import numpy as np
from PIL import Image
from eiq.multimedia.v4l2 import set_pipeline

def gstreamer_configurations(args):
    if args.webcam >= 0:
        return opencv.VideoCapture(args.webcam)

    return opencv.VideoCapture(set_pipeline(1280, 720, device=args.camera))

def resize_image(input_details, image, use_opencv: bool = False):
    _, height, width, _ = input_details[0]['shape']

    if use_opencv:
        image = opencv.resize(image, (width, height))
    else:
        image = Image.open(image).resize((width, height))

    return np.expand_dims(image, axis=0)