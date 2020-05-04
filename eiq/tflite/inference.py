import cv2 as opencv
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from eiq.utils import timeit

def get_details(interpreter):
    return interpreter.get_input_details(), interpreter.get_output_details()


def inference(interpreter):
    with timeit("Inference time"):
        interpreter.invoke()


def load_model(model):
    interpreter = Interpreter(model)
    interpreter.allocate_tensors()

    return interpreter
