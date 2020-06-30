# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.apps.switch_image.switch_image import eIQSwitchLabelImage
from eiq.apps.switch_video.switch_video import eIQVideoSwitchCore
from eiq.modules.classification.classification_tflite import eIQFireClassificationTFLite, eIQObjectClassificationTFLite
from eiq.modules.classification.classification_armnn import eIQFireClassificationArmNN
from eiq.modules.detection.facial_detection import eIQFaceAndEyesDetection, eIQFacialExpressionDetection
from eiq.modules.detection.object_detection_ssd import eIQObjectDetection, eIQObjectDetectionDNN
from eiq.modules.detection.object_detection_yolo import eIQObjectDetectionYOLOV3
from eiq.modules.detection.pose_detection import eIQPoseDetection


APPS = {'switch_image': eIQSwitchLabelImage,
        'switch_video': eIQVideoSwitchCore}

DEMOS = {'face_and_eyes_detection': eIQFaceAndEyesDetection,
         'facial_expression_detection': eIQFacialExpressionDetection,
         'fire_classification_armnn': eIQFireClassificationArmNN,
         'fire_classification_tflite': eIQFireClassificationTFLite,
         'object_classification_tflite': eIQObjectClassificationTFLite,
         'object_detection_dnn': eIQObjectDetectionDNN,
         'object_detection_tflite': eIQObjectDetection,
         'object_detection_yolov3': eIQObjectDetectionYOLOV3,
         'pose_detection': eIQPoseDetection}
