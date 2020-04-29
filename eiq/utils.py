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
from os.path import dirname
from os.path import exists
import pathlib
import requests
import shutil
import sys
from sys import stdout
import tempfile
from time import monotonic
import urllib.error
from urllib.parse import urlparse
import urllib.request
import warnings
import zipfile

from eiq import config

try:
    import progressbar
    found = True
except ImportError:
    found = False


class GoogleDriveDownloader:
    '''
    This Google Driver class was copied from Andrea Palazzi respecting its rights.
    All the modified parts below are according to MIT's LICENSE terms.
    
    Reference:
    https://github.com/ndrplz/google-drive-downloader/blob/master/LICENSE
    '''
    CHUNK_SIZE = config.CHUNK_DEFAULT_SIZE
    DOWNLOAD_URL = config.REGULAR_DOWNLOAD_URL
    @staticmethod
    def download_file_from_google_drive(
            file_id, dest_path, overwrite=False, unzip=False, showsize=False):
        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:
            session = requests.Session()
            print("Downloading {} into {}... ".format(
                file_id, dest_path), end='')
            stdout.flush()
            response = session.get(
                GoogleDriveDownloader.DOWNLOAD_URL, params={
                    'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(
                    GoogleDriveDownloader.DOWNLOAD_URL,
                    params=params,
                    stream=True)
            if showsize: print()

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(
                response, dest_path, showsize, current_download_size)
            print("done.")

            if unzip:
                try:
                    print("Unzipping...", end='')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print("done.")
                except zipfile.BadZipfile:
                    warnings.warn(
                        "Ignoring unzip since '{}' does not " \
                        "look like a valid zip file".format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(
                    GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    if showsize:
                        print("\r" + GoogleDriveDownloader.sizeof_fmt(
                                current_size[0]), end=' ')
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "{:.1f} {}{}".format(num, unit, suffix)
            num /= 1024.0
        return "{:.1f} {}{}".format(num, 'Yi', suffix)


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


def retrieve_from_url(url: str = None, name: str = None, filename: str = None):
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
        return download_url(fp, filename, url, filename_parsed.netloc)


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
                label: bool = False, epochs: bool = False):
    parser = ArgumentParser()
    if camera:
        parser.add_argument(
            '-c', '--camera', type=int, default=0,
            help="set the number your camera is identified at /dev/video<x>.")
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

    return parser.parse_args()
