# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import shutil
import sys
from setuptools import setup, find_packages

from eiq.utils import copy

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

apps_dir = os.path.join(os.getcwd(), "eiq", "apps")
base_dir = os.path.join("/opt", "eiq")
demos_dir = os.path.join(os.getcwd(), "eiq", "demos")
switch_label = "eiq/apps/switch_image/switch_image.py"
images_player = "eiq/apps/images_player/images_player.py"

install_dir_demos = os.path.join(base_dir, "demos")
install_dir_apps = os.path.join(base_dir, "apps")

if os.path.exists(base_dir):
    try:
        print("Removing {0}...".format(base_dir))
        shutil.rmtree(base_dir)
    except:
        print("shutil.rmtree() has failed" \
              "trying to remove: {}".format(base_dir))

copy(install_dir_demos, demos_dir)

try:
    pathlib.Path(install_dir_apps).mkdir(parents=True, exist_ok=True)
except:
    sys.exit("Path().mkdir() has failed" \
             "trying to create: {}".format(install_dir_apps))

shutil.copy(switch_label, install_dir_apps)
shutil.copy(images_player, install_dir_apps)

setup(name="eiq",
      version="1.0.0",
      description="A Python Framework for eIQ on i.MX Processors",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url = 'https://source.codeaurora.org/external/imxsupport/pyeiq/',
      author="Alifer Moraes, Diego Dorta, Marco Franchi",
      license="BDS-3-Clause",
      packages=find_packages(),
      zip_safe=False,
      keywords = ['ml', 'eiq', 'demos', 'apps'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Information Technology',
        'Natural Language :: English',
        'Operating System :: Other OS',
        'Programming Language :: Python :: 3.7'
      ])
