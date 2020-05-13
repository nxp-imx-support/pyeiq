# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import print_function
from argparse import ArgumentParser
from contextlib import contextmanager
from datetime import timedelta

import logging
logging.basicConfig(level=logging.INFO)

import os
from os import makedirs
from os.path import exists
import pathlib
import requests
import shutil
import sys
from sys import stdout
import tempfile
from time import monotonic
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import urllib.request

from eiq import config
from eiq.ext.google_drive_downloader import GoogleDriveDownloader

try:
    import progressbar
    found = True
except ImportError:
    found = False


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
    logging.info(" ".join("%s" % a for a in args))


def convert(elapsed):
    return str(timedelta(seconds=elapsed))


@contextmanager
def timeit(message: str = None):
    begin = monotonic()
    try:
        yield
    finally:
        end = monotonic()
        print("{0}: {1}".format(message, convert(end-begin)))


def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)


def download_url(file_path: str = None, filename: str = None,
                 url: str = None, netloc: str = None):

    if not check_connection(url):
        sys.exit("'{0}' could not be reached, " \
                 " please check your internet connection.".format(netloc))

    try:
        log("Downloading '{0}'".format(filename))
        log("From '{0}' ...".format(netloc))

        with timeit("Download time"):
            if found is True:
                urllib.request.urlretrieve(url, file_path, ProgressBar())
            else:
                urllib.request.urlretrieve(url, file_path)
    except URLError as e:
        sys.exit("Something went wrong with URLError: " % e)
    except HTTPError as e:
        sys.exit("Something went wrong with HTTPError: " % e)
    finally:
        return file_path


def retrieve_from_id(gd_id_url: str=None, pathname: str = None,
                     filename: str=None, unzip_flag: bool=False):
    dirpath = os.path.join(config.TMP_FILE_PATH, pathname)
    tmpdir = get_temporary_path(dirpath)
    if not os.path.exists(dirpath):
        try:
            pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
        except OSError:
            sys.exit("os.mkdir() function has failed: %s" % tmpdir)

    fp = os.path.join(tmpdir)
    if (os.path.isfile(fp)):
        return fp
    else:
        dst = os.path.join(tmpdir, filename)
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=gd_id_url, dest_path=dst, unzip=unzip_flag)
        return fp


def retrieve_from_url(url: str = None, name: str = None,
                      filename: str = None, unzip: bool=False):
    dirpath = os.path.join(config.TMP_FILE_PATH, name)
    if filename is None:
        filename_parsed = urlparse(url)
        filename = os.path.basename(filename_parsed.path)

    tmpdir = get_temporary_path(dirpath)
    if not os.path.exists(dirpath):
        try:
            pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
        except OSError:
            sys.exit("os.mkdir() function has failed: %s" % tmpdir)

    fp = os.path.join(tmpdir, filename)
    if (os.path.isfile(fp)):
        return fp
    else:
        file = download_url(fp, filename, url, filename_parsed.netloc)

        if unzip:
            path = os.path.dirname(file)
            shutil.unpack_archive(file, path)
            return path

        return file


def check_connection(url: str = None):
    try:
        urllib.request.urlopen(url)
        return True
    except:
        return False


def copy(target_dir, src_dir):
    if not os.path.exists(target_dir):
        try:
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            sys.exit("os.mkdir() function has failed: %s" % target_dir)

    for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)

            if os.path.isdir(file_path):
                copy(os.path.join(target_dir, file), file_path)
            else:
                if file != config.INIT_MODULE_FILE:
                    shutil.copy(file_path, target_dir)


def args_parser(camera: bool = False, webcam: bool = False,
                image: bool = False, model: bool = False,
                label: bool = False, epochs: bool = False,
                videopath: bool = False, camera_inference: bool = False):
    parser = ArgumentParser()
    if camera:
        parser.add_argument(
            '-c', '--camera', type=int, default=0,
            help="set the number your camera is identified at /dev/video<x>.")
    if camera_inference:
        parser.add_argument(
            '-ci', '--camera_inference', type=bool, default=False,
            help="set to True if you want to run inference on your camera, " \
                 "otherwise it is going to run inference on a single image.")
    if webcam:
        parser.add_argument(
            '-w', '--webcam', type=int, default=-1,
            help="if you are using a webcam, set the number your " \
                 "webcam is identified at /dev/video<x>.")
    if image:
        parser.add_argument(
            '-i', '--image', default=None,
            help="path of the image to be classified")
    if model:
        parser.add_argument(
            '-m', '--model', default=None,
            help="path of the .tflite model to be executed")
    if label:
        parser.add_argument(
            '-l', '--label', default=None,
            help="path of the file containing labels")
    if epochs:
        parser.add_argument(
            '-e', '--epochs', type=int, default=50,
            help="number of epochs for the traning")
    if videopath:
        parser.add_argument(
            '-v', '--videopath', type=None, default=0,
            help="path of the video file")

    return parser.parse_args()
