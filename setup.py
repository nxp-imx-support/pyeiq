# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import shutil
import sys
from setuptools import setup, find_packages

BASE_DIR = os.path.join(os.path.abspath(os.sep), "opt", "eiq")
DEMOS_DIR = os.path.join(os.getcwd(), "eiq", ".demos")
DEMOS_INSTALL_DIR = os.path.join(BASE_DIR, "demos")
APPS_INSTALL_DIR = os.path.join(BASE_DIR, "apps")

SWITCH_LABEL_APP = os.path.join(os.getcwd(), "eiq", "apps", "switch_image",
                                "switch_image.py")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if os.path.exists(BASE_DIR):
    try:
        print("Removing {0}...".format(BASE_DIR))
        shutil.rmtree(BASE_DIR)
    except:
        print("shutil.rmtree() has failed "
              "trying to remove: {}".format(BASE_DIR))

shutil.copytree(DEMOS_DIR, DEMOS_INSTALL_DIR)

try:
    pathlib.Path(APPS_INSTALL_DIR).mkdir(parents=True, exist_ok=True)
except:
    sys.exit("Path().mkdir() has failed "
             "trying to create: {}".format(APPS_INSTALL_DIR))

shutil.copy(SWITCH_LABEL_APP, APPS_INSTALL_DIR)

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
