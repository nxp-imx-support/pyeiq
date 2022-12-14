# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import shutil
import sys
from setuptools import setup, find_packages

from eiq.utils import copy

setup(name="eiq",
version="1.0.1",
description="eIQ package provides classes and scripts to manage the eIQ Samples Apps.",
url="",
author="Alifer Moraes <alifer.moraes@nxp.com>, Diego Dorta <diego.dorta@nxp.com>, Marco Franchi <marco.franchi@nxp.com>",
author_email="diego.dorta@nxp.com",
license="BDS-3-Clause",
packages=find_packages(),
zip_safe=False)

apps_dir = os.path.join(os.getcwd(), "eiq", "apps")
base_dir = os.path.join("/opt", "eiq")
demos_dir = os.path.join(os.getcwd(), "eiq", "demos")
switch_label = "eiq/apps/label/switch-label.py"

install_dir_demos = os.path.join(base_dir, "demos")
install_dir_apps = os.path.join(base_dir, "apps")

if os.path.exists(base_dir):
    try:
        print("Removing {0}...".format(base_dir))
        shutil.rmtree(base_dir)
    except OSError as e:
        print("Error: %s : %s" % (base_dir, e.strerror))

copy(install_dir_demos, demos_dir)

if not os.path.exists(install_dir_apps):
    try:
        pathlib.Path(install_dir_apps).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit("pathlib.Path.mkdir() function has failed: %s : %s" % (install_dir_apps, e.strerror))

    shutil.copy(switch_label, install_dir_apps)
