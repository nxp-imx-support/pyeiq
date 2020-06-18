# Copyright 2020 NXP Semiconductors
# This page was inspired on MediaPipe page: https://github.com/google/mediapipe
# SPDX-License-Identifier: BSD-3-Clause

import sphinx_rtd_theme

project = 'PyeIQ'
author = 'NXP Semiconductors'

release = 'v2.0.0'

extensions = [
    'recommonmark'
]

master_doc = 'index'

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
