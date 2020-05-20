## Copyright 2020 jrosebr1
## Copyright 2020 NXP Semiconductors
##
## This train code was copied from jrosebr1 respecting its rights. All the
## modified parts below are according to jrosebr1's LICENSE terms (MIT).
##
## SPDX-License-Identifier: BSD-3-Clause
##
## References:
## https://github.com/jrosebr1

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np

from net.firedetectionnet import FireDetectionNet
from utils import load_dataset, log
from eiq.utils import args_parser, retrieve_from_id
import config
from os.path import join


class GenerateFireDetectionModel:
    def __init__(self, epochs_num, **kwargs):
        self.__dict__.update(kwargs)
        self.epochs_num = epochs_num
        self.fire_dataset_path = " "
        self.non_fire_dataset_path = " "

        self.fire_data = " "
        self.non_fire_data = " "

        self.fire_labels = " "
        self.non_fire_labels = " "

        self.data = " "
        self.labels = " "

        self.class_totals = " "
        self.class_weight = " "

        self.model = " "
        self.aug = " "
        self.predictions = " "

        self.trainX = " "
        self.testX = " "
        self.trainY = " "
        self.testY = " "

    def retrieve(self):
        log("eIQ:", "Retrieving datasets...")
        self.fire_dataset_path = join(retrieve_from_id(config.GD_ID_FIRE_DATASET, "fire", "fire.zip", True), "fire")
        self.non_fire_dataset_path = join(retrieve_from_id(config.GD_ID_NON_FIRE_DATASET, "non-fire", "non-fire.zip", True), "non-fire")
    def loading_dataset(self):
        log("eIQ:", "Loading data...")
        self.fire_data = load_dataset(self.fire_dataset_path)
        self.non_fire_data = load_dataset(self.non_fire_dataset_path)

    def construct_classes(self):
        log("eIQ:", "Constructing class labels for the data...")
        self.fire_labels = np.ones((self.fire_data.shape[0],))
        self.non_fire_labels = np.zeros((self.non_fire_data.shape[0],))

    def stack_fire_non_fire_data(self):
        log("eIQ:",
            "Stack fire and non-fire data, scaling the data to the range [0, 1]...")
        self.data = np.vstack([self.fire_data, self.non_fire_data])
        self.labels = np.hstack([self.fire_labels, self.non_fire_labels])
        self. data /= 255

    def performe_one_hot_encoding(self):
        log("eIQ:", "Performing one-hot encoding on the labels and account...")
        self.labels = to_categorical(self.labels, num_classes=2)
        self.class_totals = self.labels.sum(axis=0)
        self.class_weight = self.class_totals.max() / self.class_totals

    def split_training_and_testing(self):
        log("eIQ:", "Constructing the training and testing split...")
        (self.trainX,
         self.testX,
         self.trainY,
         self.testY) = train_test_split(self.data,
                                        self.labels,
                                        test_size=config.TEST_SPLIT,
                                        random_state=42)

    def initiate_training(self):
        log("eIQ:", "Initializing the training data with pre-configured arguments...")
        self.aug = ImageDataGenerator(rotation_range=30,
                                      zoom_range=0.15,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.15,
                                      horizontal_flip=True,
                                      fill_mode="nearest")

    def compile_model(self):
        log("eIQ:", "Optimizing and compiling the model...")
        opt = SGD(
            lr=config.INIT_LR,
            momentum=0.9,
            decay=config.INIT_LR /
            self.epochs_num)
        self.model = FireDetectionNet.build(
            width=128, height=128, depth=3, classes=2)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])

    def train_network(self):
        log("eIQ:", "Traning the network...")
        H = self.model.fit_generator(self.aug.flow(self.trainX,
                                                   self.trainY,
                                                   batch_size=config.BATCH_SIZE),
                                     validation_data=(self.testX,
                                                      self.testY),
                                     steps_per_epoch=self.trainX.shape[0] // config.BATCH_SIZE,
                                     epochs=self.epochs_num,
                                     class_weight=self.class_weight,
                                     verbose=1)

    def evaluate_network(self):
        log("eIQ:", "Evaluating the network...")
        self.predictions = self.model.predict(
            self.testX, batch_size=config.BATCH_SIZE)

    def show_details(self):
        log("eIQ:", "Showing details of the trained model:")
        log("eIQ:", classification_report(self.testY.argmax(axis=1),
                                          self.predictions.argmax(axis=1),
                                          target_names=config.CLASSES))

    def save_model(self):
        log("eIQ:", "Serializing network...")
        self.model.save(config.MODEL_PATH_PB_FORMAT)
        self.model.save(config.MODEL_PATH_H5_FORMAT)

    def run(self):
        self.retrieve()
        self.loading_dataset()
        self.construct_classes()
        self.stack_fire_non_fire_data()
        self.performe_one_hot_encoding()
        self.split_training_and_testing()
        self.initiate_training()
        self.compile_model()
        self.train_network()
        self.evaluate_network()
        self.show_details()
        self.save_model()


if __name__ == '__main__':

    args = args_parser(epochs=True)

    fire_detection_model = GenerateFireDetectionModel(epochs_num=args.epochs)
    fire_detection_model.run()
