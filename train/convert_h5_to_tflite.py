# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf

model = tf.keras.models.load_model(
    'output/fire_detection.model/saved_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("output/fire_detection.model/converted_model.tflite", "wb").write(tflite_model)
