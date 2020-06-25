# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import timedelta

import logging

logging.basicConfig(level=logging.INFO)

from hashlib import sha1
import os
import pathlib
import requests
import shutil
from subprocess import PIPE, Popen
import sys
import tempfile
import collections
from time import monotonic
from urllib.error import URLError
from urllib.parse import urlparse
import urllib.request

from eiq.config import ID, INIT_MODULE_FILE, MAX_TIME, TMP_FILE_PATH
from eiq.helper.google_drive_downloader import GoogleDriveDownloader as gdd


class Downloader:
    def __init__(self, args):
        self.args = args
        self.downloaded_file = None

    @staticmethod
    def check_servers(url_dict):
        elapsed = {}
        min_time = MAX_TIME

        for key, val in url_dict.items():
            try:
                e_time = requests.get(val).elapsed
                elapsed[e_time] = key
            except:
                pass

        for e_time in elapsed:
            min_time = min(min_time, e_time)

        if min_time == MAX_TIME:
            return None

        return elapsed[min_time]

    @staticmethod
    def check_sha1(file_path, sha1_hash):
        with open(file_path, 'rb') as f:
            file = f.read()

        return sha1(file).hexdigest() == sha1_hash

    @staticmethod
    def download_from_url(url, filename=None, download_path=None):
        timer = Timer()

        try:
            log("Downloading '{0}'".format(filename))
            with timer.timeit("Download time"):
                urllib.request.urlretrieve(url, download_path)
        except URLError as e:
            sys.exit("Something went wrong: {}".format(e))

    def download_from_web(self, url, filename=None,
                          download_path=None, drive=False):
        if filename is None:
            filename_parsed = urlparse(url)
            filename = os.path.basename(filename_parsed.path)

        if download_path is None:
            download_path = get_temporary_path(TMP_FILE_PATH)

        try:
            pathlib.Path(download_path).mkdir(parents=True, exist_ok=True)
        except:
            sys.exit("Path().mkdir() has failed "
                     "trying to create: {}".format(download_path))

        download_path = os.path.join(download_path, filename)

        if not (os.path.exists(download_path)):
            if not drive:
                self.download_from_url(url, filename, download_path)
            else:
                try:
                    gdd.download_file_from_google_drive(file_id=url,
                                                        dest_path=download_path)
                    self.downloaded_file = download_path
                except:
                    sys.exit("Google Drive server could not be reached."
                             "Your download has been canceled.\n"
                             "Exiting...")

        self.downloaded_file = download_path

    def retrieve_data(self, url_dict, filename=None, download_path=None,
                      sha1_hash=None, unzip=False):
        if os.path.exists(os.path.join(download_path, filename)):
            return

        drive_flag = False
        if self.args.download is not None:
            if self.args.download == 'wget':
                self.wget(url_dict['github'], filename, download_path)
                return
            try:
                url = url_dict[self.args.download]
            except:
                sys.exit("Your download parameter is invalid. Exiting...")

            if self.args.download == 'drive':
                drive_flag = True
                url = url.split('/')[ID]
        else:
            print("Searching for the best server to download...")
            src = self.check_servers(url_dict)
            if src is not None:
                url = url_dict[src]
                if src == 'drive':
                    url = url.split('/')[ID]
                    drive_flag = True
            else:
                sys.exit("No servers were available to download the data.\n"
                         "Exiting...")

        self.download_from_web(url, filename, download_path, drive=drive_flag)
        if unzip and self.downloaded_file is not None:
            if sha1_hash and self.check_sha1(self.downloaded_file, sha1_hash):
                shutil.unpack_archive(self.downloaded_file, download_path)
            else:
                os.remove(self.downloaded_file)
                sys.exit("The checksum of your file failed!"
                         "Your file is corrupted.\nRemoving and exiting...")

    @staticmethod
    def wget(url, filename, download_path):
        newfile = os.path.join(download_path, filename)

        try:
            pathlib.Path(download_path).mkdir(parents=True, exist_ok=True)
        except:
            sys.exit("Path().mkdir() has failed "
                     "trying to create: {}".format(download_path))

        Popen(["wget", "{}".format(url), "-O", "{}".format(newfile)]).wait()
        shutil.unpack_archive(newfile, download_path)


def log(*args):
    logging.info(" ".join("{}".format(a) for a in args))

class Framerate:
    def __init__(self):
        self.fps = 0

    @contextmanager
    def fpsit(self, message: str = None):
        window = collections.deque(maxlen=30)
        begin = monotonic()
        try:
            yield
        finally:
            end = monotonic()
            window.append(end - begin)
            self.fps = len(window) / sum(window)

class Timer:
    def __init__(self):
        self.time = 0

    @contextmanager
    def timeit(self, message: str = None):
        begin = monotonic()
        try:
            yield
        finally:
            end = monotonic()
            self.convert(end - begin)

    def convert(self, elapsed):
        self.time = str(timedelta(seconds=elapsed))


def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)


def file_type(file_path):
    if os.path.isfile(file_path):
        try:
            proc = Popen(['file', '-bi', '{}'.format(file_path)], stdout=PIPE)
            stdout = proc.communicate()[0].decode('ascii')
            stdout = stdout.split(';')[0]
            stdout = stdout.split('/')[0]

            return stdout
        except:
            return None
    return None


def args_parser(download=False, image=False, labels=False, model=False,
                res=False, video_fwk=False, video_src=False):
    parser = ArgumentParser()
    if download:
        parser.add_argument(
            '-d', '--download', default=None,
            help="Choose from which server the models are going to be "
                 "downloaded")
    if image:
        parser.add_argument(
            '-i', '--image', default=None,
            help="path of the image to be classified")
    if labels:
        parser.add_argument(
            '-l', '--labels', default=None,
            help="path of the file containing labels")
    if model:
        parser.add_argument(
            '-m', '--model', default=None,
            help="path of the .tflite model to be executed")
    if res:
        parser.add_argument(
            '-r', '--res', default='hd',
            help="Choose the video capture device resolution as bellow:"
                 "full_hd: 1920x1080. "
                 "hd: 1280x720. "
                 "vga: 640x480. "
                 "Default resolution is hd, if supported, else it uses the "
                 "best supported resolution.")
    if video_fwk:
        parser.add_argument(
            '-f', '--video_fwk', default='v4l2',
            help="Choose the video framework according to the options below:"
                 "v4l2: GStreamer appsink + OpenCV.imshow(). Default."
                 "opencv: OpenCV.VideoCapture() + OpenCV.imxshow()."
                 "gstreamer: GStreamer appsink + GStreamer appsrc.")
    if video_src:
        parser.add_argument(
            '-v', '--video_src', default=None,
            help="Choose your video source, it can be the path to a video file"
                 " or your video device, e.g, /dev/video<x>")

    return parser.parse_args()
