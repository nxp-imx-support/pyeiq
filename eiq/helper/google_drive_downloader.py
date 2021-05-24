# Copyright 2017 Andrea Palazzi 
# Copyright 2021 NXP Semiconductors
#
# This Google Driver class was copied from Andrea Palazzi respecting its rights.
# All the modified parts below are according to MIT's LICENSE terms.
#
# SPDX-License-Identifier: MIT
# Reference:
# https://github.com/ndrplz/google-drive-downloader/blob/master/LICENSE

from __future__ import print_function
import zipfile
import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists

import requests

from eiq import config

class GoogleDriveDownloader:
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
