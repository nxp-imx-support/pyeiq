# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.apps.switch_image.switch_image import eIQSwitchLabelImage
from eiq.apps.switch_video.switch_video import eIQVideoSwitchCore
from eiq.modules.classification.classification_tflite import eIQObjectClassificationTFLite
from eiq.modules.detection.object_detection_ssd import eIQObjectDetection
from eiq.modules.detection.covid19_detection import eIQCOVID19Detection


APPS = {'switch_image': eIQSwitchLabelImage,
        'switch_video': eIQVideoSwitchCore}

DEMOS = {'covid19_detection': eIQCOVID19Detection,
         'object_classification_tflite': eIQObjectClassificationTFLite,
         'object_detection_tflite': eIQObjectDetection
         }
