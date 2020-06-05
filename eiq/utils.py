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
from os import makedirs
from os.path import exists
import pathlib
import requests
import shutil
import subprocess
import sys
from sys import stdout
import tempfile
from time import monotonic
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import urllib.request

from eiq.config import *
from eiq.helper.google_drive_downloader import GoogleDriveDownloader as gdd

try:
    import progressbar
    found = True
except ImportError:
    found = False


class Downloader():
    def __init__(self, args):
        self.args = args
        self.downloaded_file = None

    def check_servers(self, url_dict):
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

    def check_sha1(self, file_path, sha1_hash):
        with open(file_path, 'rb') as f:
            file = f.read()

        return sha1(file).hexdigest() == sha1_hash

    def download_from_url(self, url, filename=None, download_path=None):
        timer = InferenceTimer()

        try:
            log("Downloading '{0}'".format(filename))
            with timer.timeit("Download time"):
                if found is True:
                    urllib.request.urlretrieve(url, download_path,
                                               ProgressBar())
                else:
                    urllib.request.urlretrieve(url, download_path)
        except URLError as e:
            sys.exit("Something went wrong with URLError: {}".format(e))
        except HTTPError as e:
            sys.exit("Something went wrong with HTTPError: {}".format(e))

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
            sys.exit("Path().mkdir() has failed" \
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
                    sys.exit("Google Drive server could not be reached." \
                             "Your download has been canceled.\n" \
                             "Exiting...")

        self.downloaded_file = download_path

    def retrieve_data(self, url_dict, filename=None, download_path=None,
                      sha1=None, unzip=False):
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
                sys.exit("No servers were available to download the data.\n" \
                         "Exiting...")

        self.download_from_web(url, filename, download_path, drive=drive_flag)
        if unzip and self.downloaded_file is not None:
            if sha1 is not None and self.check_sha1(self.downloaded_file,
                                                    sha1):
                shutil.unpack_archive(self.downloaded_file, download_path)
            else:
                os.remove(self.downloaded_file)
                sys.exit("The checksum of your file failed!"\
                         "Your file is corrupted.\nRemoving and exiting...")

    def wget(self, url, filename, download_path):
        file = os.path.basename(url)
        newfile = os.path.join(download_path, filename)
        proc = subprocess.Popen("mkdir -p {}".format(download_path), shell = True)
        proc.wait()
        proc = subprocess.Popen("wget {}".format(url), shell = True)
        proc.wait()
        proc =subprocess.Popen("mv {} {}".format(file, newfile), shell = True)
        proc.wait()

        shutil.unpack_archive(newfile, download_path)


class ProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def log(*args):
    logging.info(" ".join("{}".format(a) for a in args))


class InferenceTimer:
    def __init__(self):
        self.time = 0

    @contextmanager
    def timeit(self, message: str = None):
        begin = monotonic()
        try:
            yield
        finally:
            end = monotonic()
            self.convert(end-begin)
            print("{0}: {1}".format(message, self.time))

    def convert(self, elapsed):
        self.time = str(timedelta(seconds=elapsed))


def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)


def copy(target_dir, src_dir):
    try:
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    except:
        sys.exit("Path().mkdir() has failed" \
                 "trying to create: {}".format(target_dir))

    for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)

            if os.path.isdir(file_path):
                copy(os.path.join(target_dir, file), file_path)
            else:
                if file != INIT_MODULE_FILE:
                    shutil.copy(file_path, target_dir)


def args_parser(camera_inference=False, download=False, epochs=False,
                image=False, label=False, model=False, video_src=False,
                video_fwk=False):
    parser = ArgumentParser()
    if camera_inference:
        parser.add_argument(
            '-ci', '--camera_inference', type=bool, default=False,
            help="set to True if you want to run inference on your camera, " \
                 "otherwise it is going to run inference on a single image.")
    if download:
        parser.add_argument(
            '-d', '--download', default=None,
            help="Choose from which server the models are going to be " \
                 "downloaded")
    if epochs:
        parser.add_argument(
            '-e', '--epochs', type=int, default=50,
            help="number of epochs for the traning")
    if image:
        parser.add_argument(
            '-i', '--image', default=None,
            help="path of the image to be classified")
    if label:
        parser.add_argument(
            '-l', '--label', default=None,
            help="path of the file containing labels")
    if model:
        parser.add_argument(
            '-m', '--model', default=None,
            help="path of the .tflite model to be executed")
    if video_fwk:
        parser.add_argument(
            '-f', '--video_fwk', default=None,
            help="Choose the video framework between v4l2, gstreamer and overlay")
    if video_src:
        parser.add_argument(
            '-v', '--video_src', default=None,
            help="Choose your video source, it can be the path to a video file" \
                 " or your video device, e.g, /dev/video<x>")

    return parser.parse_args()
