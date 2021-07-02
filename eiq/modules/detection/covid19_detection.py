# Copyright 2020 PureHing
# Copyright 2021 NXP Semiconductors
#
# Parts of this file were copied from PureHing's github repository
# respecting its rights. All the modified parts below are according to
# MIT LICENSE terms.
#
# SPDX-License-Identifier:    MIT
# Reference:
# https://github.com/PureHing/face-mask-detection-tf2/blob/master/LICENSE

import os
import cv2
import numpy as np
from PIL import Image

from eiq.config import FONT
from eiq.engines.tflite.inference import TFLiteInterpreter
from eiq.modules.detection.config import LEFT, TOP, RIGHT, BOTTOM, CONFIDENCE, CLASSES, OBJ_COVID19
from eiq.modules.utils import DemoBase


class eIQCOVID19Detection(DemoBase):
    def __init__(self, args=None):
        super().__init__(args, self.__class__.__name__, OBJ_COVID19)

        self.social_distance_model = None
        self.facemask_model = None
        self.interpreter_social_distance = None
        self.interpreter_facemask = None

        self.mode = None

        #camera info
        self.eye_height = None
        self.eye_angle = None
        self.eye_slope = None

        self.label_person =  ['???','person']
        #facemask
        self.labels_facemask =  {0: 'Mask', 1: 'NoMask'}
        self.priors = None
        self.count = 0

        self.run_ssd = 1
        self.run_facemask = 0
        self.result_ssd = None
        self.result_face = None

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * Single Shot Detection as default algorithm.\n")

    def prepare_params(self):
        flag = None

        if hasattr(self.args, 'work_mode'):
            flag = self.args.work_mode

        priors_file = os.path.join(self.model_dir, 'priors.txt')
        self.priors = np.reshape(np.loadtxt(priors_file), (9949, 4))

        #set self.model
        self.social_distance_model = os.path.join(self.model_dir, "mobilenet_ssd_v2_coco_quant_postprocess.tflite")
        self.facemask_model = os.path.join(self.model_dir, "facemask_int8.tflite")
        #set cam param
        camera_params = []
        if hasattr(self.args, 'camera_params'):
            camera_params = self.args.camera_params.split()
            self.eye_height = int(camera_params[0])
            self.eye_angle = int(camera_params[1])
            self.eye_slope = int(camera_params[2])
        else:
            self.eye_height = 200
            self.eye_angle = 60
            self.eye_slope = 60

        """
        SSD FACEMASK IMAGE VIDEO CAMERA

        """
        if flag == 'None':
            self.mode = 0
        elif flag == '1':
            self.mode = 1
        else :
            self.mode = 0

    def usage(self, name=None, labels=True, model=True):
        super().usage(name=name, labels=labels, model=model)


    def softmax(self, x):
        orig_shape = x.shape
        #print("orig_shape", orig_shape)

        if len(x.shape) > 1:
            # matrix
            tmp = np.max(x, axis=1)
            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis=1)
            x /= tmp.reshape((x.shape[0], 1))
            #print("matrix")
        else:
            # vector
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
            #print("vector")
        return x

    def social_distant_infer(self, image):
        self.interpreter_social_distance.set_tensor(np.expand_dims(image, axis=0))
        self.interpreter_social_distance.run_inference()

        positions = self.interpreter_social_distance.get_tensor(0, squeeze=True)
        classes = self.interpreter_social_distance.get_tensor(1, squeeze=True)
        scores = self.interpreter_social_distance.get_tensor(2, squeeze=True)

        classes = np.squeeze(classes+1).astype(np.int32)

        result = []
        for idx, score in enumerate(scores):
            if (score > 0.5) and (classes[idx] == 1):
                result.append({'pos': positions[idx], '_id': classes[idx]})
        return result

    def draw_result(self, frame, result):
        width = frame.shape[1]
        height = frame.shape[0]

        GPS = []

        for obj in result:
            pos = obj['pos']
            _id = obj['_id']

            x1 = int(pos[1] * width)
            x2 = int(pos[3] * width)
            y1 = int(pos[0] * height)
            y2 = int(pos[2] * height)

            top = max(0, np.floor(y1 + 0.5).astype('int32'))
            left = max(0, np.floor(x1 + 0.5).astype('int32'))
            bottom = min(height, np.floor(y2 + 0.5).astype('int32'))
            right = min(width, np.floor(x2 + 0.5).astype('int32'))

            #method 1, not calculate > 90° range
            angle_y =(1 - bottom / height) * self.eye_angle + (self.eye_slope - self.eye_angle / 2)
            if (angle_y >= 90):
                continue

            angle_x = ((left + right) / 2 / width - 0.5) * (self.eye_angle / 2)
            Y = self.eye_height * np.tan(angle_y * np.pi / 180)
            X = Y * np.tan(angle_x * np.pi / 180) * 1.4
            GPS.append({'top':top, 'left': left, 'bottom':bottom,'right':right, 'X':X, 'Y':Y, 'warning': False})
            # end method 1

        return GPS

    def draw_result_ssd(self, frame, GPS):

        if GPS == None:
            return
        #calculate social distance
        persons = len(GPS)
        for index_1 in range(persons-1):
            for j  in range(persons - index_1 - 1):
                index_2 = index_1 + j + 1
                if abs(GPS[index_2]['X'] - GPS[index_1]['X']) < 100 and abs(GPS[index_2]['Y'] - GPS[index_1]['Y']) < 100:
                    rect1_x = (int)((GPS[index_1]['right'] + GPS[index_1]['left']) / 2)
                    rect1_y = (int)((GPS[index_1]['top'] + GPS[index_1]['bottom']) / 2)
                    rect2_x = (int)((GPS[index_2]['right'] + GPS[index_2]['left']) / 2)
                    rect2_y = (int)((GPS[index_2]['top'] + GPS[index_2]['bottom']) / 2)

                    GPS[index_1]['warning'] = True
                    GPS[index_2]['warning'] = True

                    cv2.line(frame,(rect1_x,rect1_y),(rect2_x,rect2_y),(0,0,255), 3)

        for index in range(persons):
            if GPS[index]['warning'] == True:
                color = FONT['color']['red']
            else:
                color = FONT['color']['green']

            left = (int)(GPS[index]['left'])
            top  = (int)(GPS[index]['top'])
            right = (int)(GPS[index]['right'])
            bottom  = (int)(GPS[index]['bottom'])

            cv2.rectangle(frame, (left, top), (right, bottom),
                          color, 6)


        self.overlay.draw_info(frame, self.social_distance_model, self.media_src,
                                   self.interpreter_social_distance.inference_time,
                                   self.interpreter_social_distance.input_details[0]['quantization'])

    def decode_bbox(self, bbox, priors, variances):
        """
        Decode locations from predictions using anchors to undo
        the encoding we did for offset regression at train time.
        """
        if variances is None:
            variances = [0.1, 0.2]
        boxes = np.concatenate(
            (priors[:, :2] + bbox[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(bbox[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def single_class_non_max_suppression(self, bboxes, confidences, conf_thresh=0.6, iou_thresh=0.5, keep_top_k=-1):
        '''
        do nms on single class.
        Hint: for the specific class, given the bbox and its confidence,
        1) sort the bbox according to the confidence from top to down, we call this a set
        2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
        3) remove the bbox whose IOU is higher than the iou_thresh from the set,
        4) loop step 2 and 3, util the set is empty.
        :param bboxes: numpy array of 2D, [num_bboxes, 4]
        :param confidences: numpy array of 1D. [num_bboxes]
        :param conf_thresh:
        :param iou_thresh:
        :param keep_top_k:
        :return:
        '''
        if len(bboxes) == 0: return []

        conf_keep_idx = np.where(confidences > conf_thresh)[0]

        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]

        pick = []
        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)

        # if the number of final bboxes is less than keep_top_k, we need to pad it.
        # TODO
        return conf_keep_idx[pick]


    def facemask_infer(self, input_data, image_pad, image_raw):

        height_pad, width_pad, _ = image_pad.shape
        height, width, _ = image_raw.shape
        variances = [0.1, 0.2]
        result = []

        self.interpreter_facemask.set_tensor(np.expand_dims(input_data, axis=0))
        self.interpreter_facemask.run_inference()

        predictions = self.interpreter_facemask.get_tensor(0, squeeze=False)


        bbox_regressions, confs = np.split(predictions[0], [4, ], axis=-1)
        boxes = self.decode_bbox(bbox_regressions, self.priors, variances)

        confs = self.softmax(confs)
        confs = confs[:, [1, 2]]

        bbox_max_scores = np.max(confs, axis=1)
        bbox_max_score_classes = np.argmax(confs, axis=1)

        keep_idxs = self.single_class_non_max_suppression(boxes,
                                                 bbox_max_scores,
                                                 conf_thresh=0.5,
                                                 iou_thresh=0.5,
                                                 )

        ######### draw image
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = boxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width_pad))
            ymin = max(0, int(bbox[1] * height_pad))
            xmax = min(int(bbox[2] * width_pad), width)
            ymax = min(int(bbox[3] * height_pad), height)

            result.append([xmin,ymin,xmax,ymax,class_id])

        return result

    def draw_result_face(self, frame, result):

        if result == None:
            return

        for idx in result:
            if idx[4] == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (idx[0], idx[1]), (idx[2], idx[3]), color, 2)
            cv2.putText(frame, "%s" % (self.labels_facemask[idx[4]]), (idx[0] + 2, idx[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

        self.overlay.draw_info(frame, self.facemask_model, self.media_src,
                                   self.interpreter_facemask.inference_time,
                                   self.interpreter_facemask.input_details[0]['quantization'])

    def pad_input_image(self, img):
        """pad image to suitable shape"""
        img_h, img_w, _ = img.shape

        img_pad_h = 0
        img_pad_w = 0

        if img_w > img_h:
            img_pad_h = img_w - img_h
        else:
            img_pad_w = img_h - img_w

        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT)
        return img

    def image_pre_process(self, image_raw):

        img_f = np.float32(image_raw.copy())
        img_pad = self.pad_input_image(img_f)
        img_resize = cv2.resize(img_pad,(self.interpreter_facemask.width(), self.interpreter_facemask.height()))
        img_data =  img_resize / 255.0 - 0.5
        img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        return img, img_pad

    def detect_object(self, frame):


        if self.mode == 1:
            self.count += 1
            self.run_ssd = self.count % 2
            self.run_facemask = (self.count + 1) % 2
        else :
            self.run_ssd = 0
            self.run_facemask = 1

        if self.run_ssd:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((self.interpreter_social_distance.width(), self.interpreter_social_distance.height()))
            result = self.social_distant_infer(image)
            self.result_ssd = self.draw_result(frame, result)

        if self.run_facemask:
            facemask_input, facemask_pad = self.image_pre_process(frame)
            self.result_face = self.facemask_infer(facemask_input, facemask_pad, frame)

        self.draw_result_ssd(frame, self.result_ssd)
        self.draw_result_face(frame, self.result_face)


        if self.count == 1000:
            self.count = 0

        return frame

    def start(self):
        self.gather_data()
        self.prepare_params()
        self.interpreter_social_distance = TFLiteInterpreter(self.social_distance_model)
        self.interpreter_facemask = TFLiteInterpreter(self.facemask_model)


    def run(self):
        self.start()
        self.run_inference(self.detect_object)
