import os
from setuptools import setup, find_packages
from eiq.utils import copy

setup(name="eiq",
version="1.0",
description="eIQ package provides classes and scripts to manage the eIQ Samples Apps.",
url="",
author="Diego Dorta <diego.dorta@nxp.com>, Alifer Moraes <alifer.moraes@nxp.com>",
author_email="diego.dorta@nxp.com",
license="BDS-3-Clause",
packages=find_packages(),
zip_safe=False)

demos_dir = os.path.join(os.getcwd(), "eiq", "demos")
apps_dir = os.path.join(os.getcwd(), "eiq", "apps")
install_dir_base = os.path.join("/opt", "eiq")

install_dir_demos = os.path.join(install_dir_base, "demos")
install_dir_apps = os.path.join(install_dir_base, "apps")

copy(install_dir_demos, demos_dir)
copy(install_dir_apps, apps_dir)