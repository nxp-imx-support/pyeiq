from imutils import paths
import os
import tempfile
import pathlib
import logging
logging.basicConfig(level = logging.INFO)

import cv2
import numpy as np

import config
from helper.google_driver_downloader import GoogleDriveDownloader as gdd

def log(*args):
    logging.info(" ".join("%s" %a for a in args))

def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)

def retrieve_from_id(gd_id_url: str = None, filename: str = None, unzip_flag: bool = False):
    name = "fire_detection"
    dirpath = os.path.join(config.TMP_FILE_PATH, name)
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
        try:
            dst = os.path.join(tmpdir, filename + '.zip')
            gdd.download_file_from_google_drive(file_id=gd_id_url, dest_path=dst, unzip=unzip_flag)
        except ImportError:
            sys.exit("Could not find GoogleDriverDownloader Module")
        finally:
            return fp
    return 

def load_dataset(dataset_path):
	image_paths = list(paths.list_images(dataset_path))
	data = []

	for image_path in image_paths:
		image = cv2.imread(image_path)
		image = cv2.resize(image, (128, 128))
		data.append(image)
	return np.array(data, dtype="float32")
