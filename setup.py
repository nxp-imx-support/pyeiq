# Copyright 2021 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import os
from setuptools import setup, find_packages

PYEIQ_LAUNCHER = os.path.join(os.getcwd(), "eiq", "apps", "pyeiq_launcher", "pyeiq.py")
PYEIQ_USR = os.path.join("/usr", "bin", "pyeiq")

if os.path.exists(PYEIQ_USR):
    os.system("rm -rf {}".format(PYEIQ_USR))

os.system("cp {} {}".format(PYEIQ_LAUNCHER, PYEIQ_USR))

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name="pyeiq",
      version="3.0.0",
      description="A Python Framework for eIQ on i.MX Processors",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url = 'https://source.codeaurora.org/external/imxsupport/pyeiq/',
      author="Alifer Moraes, Diego Dorta, Marco Franchi, Kaixin Ding, Gnar Fang, Michael Pontikes, Jinghui Liu, Manish Bajaj, Devin Jiao",
      license="BSD-3-Clause",
      packages=find_packages(),
      zip_safe=False,
      install_requires=['requests'],
      keywords = ['ml', 'eiq', 'demos', 'apps'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Information Technology',
        'Natural Language :: English',
        'Operating System :: Other OS',
        'Programming Language :: Python :: 3.9'
      ])
