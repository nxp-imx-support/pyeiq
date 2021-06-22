# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
from socket import gethostname
from stat import S_IEXEC
import subprocess
import threading
import time

import cv2
import numpy as np
from PIL import Image

from eiq.apps.switch_video.config import *
from eiq.apps.utils import create_link
from eiq.config import BASE_DIR,ZIP
from eiq.utils import args_parser, check_data, Downloader, file_type


class eIQVideoSwitchCore:
    def __init__(self):
        self.args = args_parser()
        self.pid = [0, 0, 0]
        self.device = None
        self.class_name = self.__class__.__name__
        self.data = VIDEO_SWITCH_CORE

        self.base_dir = os.path.join(BASE_DIR,self.class_name)
        self.binary = os.path.join(self.base_dir, "bin", "video_switch_core")
        self.labels = os.path.join(self.base_dir, "model", "labels.txt")
        self.media = os.path.join(self.base_dir, "media", "video_device.mp4")
        self.media_backup = os.path.join(self.base_dir, "media", "backup.mp4")
        self.model = os.path.join(self.base_dir, "model", "ssd_mobilenet_v1_1_default_1.tflite")
        self.tmp_img = NPU_IMG
        
        self.backup()

    @staticmethod
    def description():
        return ("This demo uses:\n   * TensorFlow Lite as inference engine."
                "\n   * Single Shot Detection as default algorithm.\n\n"
                "This application allows users to run an object detection demo\n"
                "using CPU, GPU or NPU to perform inference on a video.\n")
    
    def gather_data(self):
        downloader = Downloader(self.args)
        downloader.retrieve_data(self.data['src'], self.class_name+ZIP,
                                 self.base_dir, self.data['sha1'], True)
    
    def run_inference(self, device):
        self.pid[device] = subprocess.Popen(RUN.format(device, self.binary, device), shell=True).pid

    def get_device(self):
        hostname = gethostname()

        if "imx8mp" in hostname:
            self.device = "npu"
        elif "imx8" in hostname:
            self.device = "gpu"            

    def pause_proc(self, dev):
        if self.pid[dev] > 0:
            proc = subprocess.Popen(PAUSE.format(self.pid[dev]), shell=True)
            proc.wait()

    def resume_proc(self, dev):
        if self.pid[dev] > 0:
            proc = subprocess.Popen(RESUME.format(self.pid[dev]), shell=True)
            proc.wait()

    def interruption(self):
        self.get_device()

        while True:
            interrupt = str(input("Choose: 'cpu' 'gpu' '{}': ".format(self.device)))
            os.system("clear")

            if interrupt == "cpu":
                self.pause_proc(GPU)
                self.pause_proc(NPU)
                self.resume_proc(CPU)
                self.tmp_img = CPU_IMG
            elif interrupt == "gpu":                
                self.pause_proc(CPU)
                self.pause_proc(NPU)
                self.resume_proc(GPU)
                self.tmp_img = GPU_IMG
            elif interrupt == "npu":                
                self.pause_proc(CPU)
                self.pause_proc(GPU)
                self.resume_proc(NPU)
                self.tmp_img = NPU_IMG
            else:
                print("Invalid option. Please, choose between 'cpu' 'gpu' '{}': ".format(self.device))

    def start_threads(self):
        b = np.zeros([480, 640, 3], 'uint8')
        b[:,:,0] = 255
        b[:,:,1] = 255
        b[:,:,2] = 255
        img = Image.fromarray(b)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(CPU_IMG, cv_img)
        cv2.imwrite(NPU_IMG, cv_img)
        cv2.imwrite(GPU_IMG, cv_img)

        input_thread = threading.Thread(target=self.interruption)
        
        cpu_inference_thread = threading.Thread(target=self.run_inference,
                                                args=(CPU, ))
        npu_inference_thread = threading.Thread(target=self.run_inference,
                                                args=(GPU, ))
        gpu_inference_thread = threading.Thread(target=self.run_inference,
                                              args=(NPU, ))
                                              
        
        input_thread.daemon = True
        cpu_inference_thread.daemon = True
        npu_inference_thread.daemon = True
        gpu_inference_thread.daemon = True
        npu_inference_thread.start()
        time.sleep(5)
        gpu_inference_thread.start()
        time.sleep(5)
        cpu_inference_thread.start()
        time.sleep(3)
        input_thread.start()

    def backup(self):
        if os.path.isfile(self.media_backup):
            os.system(f"cp {self.media_backup} {self.media}")

    def start(self):
        os.environ['VSI_NN_LOG_LEVEL'] = "0"
        if not check_data(os.path.join(self.base_dir, self.class_name + ZIP),
                          self.data['sha1'], self.binary, self.labels,
                          self.media, self.model):
                          
            self.gather_data()
        os.chmod(self.binary, S_IEXEC)
        create_link()
        
        if self.args.video_src and os.path.isfile(self.args.video_src):
            if file_type(self.args.video_src) != "video":
                print("Your video_src is not a valid video file. Using the default video...")
            else:
                os.system(f"cp {self.media} {self.media_backup}")
                os.system(f"cp {self.args.video_src} {self.media}")
        else:
            print("Your video_src is not a valid video file. Using the default video...")

        self.start_threads()

    def run(self):
        self.start()

        while True:
            with open(self.tmp_img, 'rb') as f:
                bytes_img = f.read()

            if bytes_img.endswith(JPEG_EOF):
                image = cv2.imdecode(np.frombuffer(bytes_img, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                cv2.imshow("window_title", image)
                cv2.waitKey(5)
