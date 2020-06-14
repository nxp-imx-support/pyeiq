# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os
import tempfile

TMP_FILE_PATH = "eiq"

BASE_DIR = os.path.join(os.path.abspath(os.sep), tempfile.gettempdir(), TMP_FILE_PATH)

CHUNK_DEFAULT_SIZE = 32768

REGULAR_DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

INIT_MODULE_FILE = "__init__.py"

MAX_TIME = datetime.timedelta(9, 9, 9)

ID = 5

ZIP = ".zip"
