from enum import Enum
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from eiq.config import BASE_DIR
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.utils import real_time_inference
from eiq.posenet.config import *
from eiq.utils import args_parser, Downloader

class eIQCoralPoseNet:
	def __init__(self):
		self.args = args_parser(download=True, image=True, model=True,
								video_fwk=True, video_src=True)
		self.base_dir = os.path.join(BASE_DIR, self.__class__.__name__)
		self.media_dir = os.path.join(self.base_dir, "media")
		self.model_dir = os.path.join(self.base_dir, "model")

		self.image = None
		self.model = None

		self.input_mean = 127.5
		self.input_std = 127.5
		self.min_confidence = 0.4

	class BodyPart(Enum):
		NOSE = 0,
		LEFT_EYE = 1,
		RIGHT_EYE = 2,
		LEFT_EAR = 3,
		RIGHT_EAR = 4,
		LEFT_SHOULDER = 5,
		RIGHT_SHOULDER = 6,
		LEFT_ELBOW = 7,
		RIGHT_ELBOW = 8,
		LEFT_WRIST = 9,
		RIGHT_WRIST = 10,
		LEFT_HIP = 11,
		RIGHT_HIP = 12,
		LEFT_KNEE = 13,
		RIGHT_KNEE = 14,
		LEFT_ANKLE = 15,
		RIGHT_ANKLE = 16,

	class Position:
		def __init__(self):
			self.x = 0
			self.y = 0

	class KeyPoint:
		def __init__(self):
			self.bodyPart = self.BodyPart.NOSE
			self.position = self.Position()
			self.score = 0.0

	class Person:
		def __init__(self):
			self.keyPoints = []
			self.score = 0.0

	def gather_data(self):
		download = Downloader(self.args)
		download.retrieve_data(CORAL_POSENET_MODEL_SRC,
							   self.__class__.__name__ + ZIP,self.base_dir,
							   CORAL_POSENET_SHA1, True)
		self.model = os.path.join(self.model_dir,
								  CORAL_POSENET_MODEL_NAME)

		if self.args.image is not None and os.path.exists(self.args.image):
			self.image = self.args.image
		else:
			self.image = os.path.join(self.media_dir,
									  CORAL_POSENET_MEDIA_NAME)

	def sigmoid(self, x):
		return 1. / (1. + math.exp(-x))

	def estimate_pose(self, image):
		w, h = image.size
		image = image.resize((self.interpreter.width(),
		                      self.interpreter.height()))
		image = np.expand_dims(image, axis=0)

		if self.interpreter.dtype() == np.float32:
			image = (np.float32(image) - self.input_mean) / self.input_std

		self.interpreter.set_tensor(image)
		self.interpreter.run_inference()

		heat_maps = self.interpreter.get_tensor(0)
		offset_maps = self.interpreter.get_tensor(1)

		height = len(heat_maps[0])
		width = len(heat_maps[0][0])
		num_key_points = len(heat_maps[0][0][0])
		key_point_positions = [[0] * 2 for i in range(num_key_points)]

		for key_point in range(num_key_points):
			max_val = heat_maps[0][0][0][key_point]
			max_row = 0
			max_col = 0
			for row in range(height):
				for col in range(width):
					heat_maps[0][row][col][key_point] = self.sigmoid(heat_maps[0][row][col][key_point])
					if heat_maps[0][row][col][key_point] > max_val:
						max_val = heat_maps[0][row][col][key_point]
						max_row = row
						max_col = col
			key_point_positions[key_point] = [max_row, max_col]

		x_coords = [0] * num_key_points
		y_coords = [0] * num_key_points
		confidenceScores = [0] * num_key_points

		for i, position in enumerate(key_point_positions):
			position_y = int(key_point_positions[i][0])
			position_x = int(key_point_positions[i][1])
			y_coords[i] = (position[0]/float(height-1) * h \
						  + offset_maps[0][position_y][position_x][i])
			x_coords[i] = (position[1]/float(width-1) * w \
						  + offset_maps[0][position_y][position_x][i + num_key_points])
			confidenceScores[i] = heat_maps[0][position_y][position_x][i]

		person = self.Person()
		key_point_list = []
		total_score = 0

		for i in range(num_key_points):
			key_point = self.KeyPoint()
			key_point_list.append(key_point)

		for i, body_part in enumerate(self.BodyPart):
			key_point_list[i].bodyPart = body_part
			key_point_list[i].position.x = x_coords[i]
			key_point_list[i].position.y = y_coords[i]
			key_point_list[i].score = confidenceScores[i]
			total_score += confidenceScores[i]

		person.keyPoints = key_point_list
		person.score = total_score / num_key_points

		return person

	def detect_pose(self, frame):
		frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		draw = ImageDraw.Draw(frame)
		person = self.estimate_pose(frame)

		body_joints = [[self.BodyPart.LEFT_WRIST, self.BodyPart.LEFT_ELBOW],
					   [self.BodyPart.LEFT_ELBOW, self.BodyPart.LEFT_SHOULDER],
					   [self.BodyPart.LEFT_SHOULDER, self.BodyPart.RIGHT_SHOULDER],
					   [self.BodyPart.RIGHT_SHOULDER, self.BodyPart.RIGHT_ELBOW],
					   [self.BodyPart.RIGHT_ELBOW, self.BodyPart.RIGHT_WRIST],
					   [self.BodyPart.LEFT_SHOULDER, self.BodyPart.LEFT_HIP],
					   [self.BodyPart.LEFT_HIP, self.BodyPart.RIGHT_HIP],
					   [self.BodyPart.RIGHT_HIP, self.BodyPart.RIGHT_SHOULDER],
					   [self.BodyPart.LEFT_HIP, self.BodyPart.LEFT_KNEE],
					   [self.BodyPart.LEFT_KNEE, self.BodyPart.LEFT_ANKLE],
					   [self.BodyPart.RIGHT_HIP, self.BodyPart.RIGHT_KNEE],
					   [self.BodyPart.RIGHT_KNEE, self.BodyPart.RIGHT_ANKLE]]

		for line in body_joints:
			if person.keyPoints[line[0].value[0]].score > self.min_confidence \
			and person.keyPoints[line[1].value[0]].score > self.min_confidence:
				start_point_x = int(person.keyPoints[line[0].value[0]].position.x)
				start_point_y = int(person.keyPoints[line[0].value[0]].position.y)
				end_point_x = int(person.keyPoints[line[1].value[0]].position.x)
				end_point_y = int(person.keyPoints[line[1].value[0]].position.y)
				draw.line((start_point_x, start_point_y, end_point_x, end_point_y),
						  fill=(255, 255, 0), width=3)

		for key_point in person.keyPoints:
			if key_point.score > self.min_confidence:
				left_top_x = int(key_point.position.x) - 5
				left_top_y = int(key_point.position.y) - 5
				right_bottom_x = int(key_point.position.x) + 5
				right_bottom_y = int(key_point.position.y) + 5
				draw.ellipse((left_top_x, left_top_y, right_bottom_x, right_bottom_y),
							 fill=(0, 128, 0), outline=(255, 255, 0))

		frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
		cv2.imshow(TITLE_CORAL_POSENET, frame)

	def start(self):
		os.environ['VSI_NN_LOG_LEVEL'] = "0"
		self.gather_data()
		self.interpreter = TFLiteInterpreter(self.model)

	def run(self):
		self.start()

		if self.args.video_src:
			real_time_inference(self.detect_pose, self.args)
		else:
			frame = cv2.imread(self.image, cv2.IMREAD_COLOR)
			self.detect_pose(frame)
			cv2.waitKey()
		cv2.destroyAllWindows()
