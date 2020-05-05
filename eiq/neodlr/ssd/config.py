from os.path import join

TMP_DIR = join('/tmp', 'eiq')

OBJ_DETECTION_DLR_MODEL_ID = "16gQBOA509-rCB8Mw6ltzoKid96epjMaO"
OBJ_DETECTION_DLR_MODEL_NAME = "model.zip"

WIDTH = 1280
HEIGHT = 720

NN_IN_SIZE = 128
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = [(0xFF,0x83,0x00),(0xFF,0x66,0x00),(0xFF,0x00,0x00),(0x99,0xFF,0x00),
          (0x00,0xFF,0x00),(0x00,0x00,0xFF),(0x00,0x00,0x00)]

# Mean and Std deviation of the RGB colors (collected from Imagenet dataset)
MEAN = [123.68,116.779,103.939]
STD = [58.393,57.12,57.375]