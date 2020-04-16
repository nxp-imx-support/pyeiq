from contextlib import contextmanager
from datetime import timedelta
from time import monotonic
from urllib.parse import urlparse
import argparse
import os
import pathlib
import sys
import tempfile
import urllib.request
import urllib.error
import logging
logging.basicConfig(level = logging.INFO)

from eiq import config

try:
    import progressbar
    found = True
except ImportError:
    found = False

class ProgressBar():
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
    logging.info(" ".join("%s" %a for a in args))

def convert(elapsed):
    return str(timedelta(seconds=elapsed))


@contextmanager
def timeit(message: str = None):
    begin = monotonic()
    try:
        yield
    finally:
        end = monotonic()
        print('{0}: {1}'.format(message, convert(end-begin)))


def get_temporary_path(*path):
    return os.path.join(tempfile.gettempdir(), *path)


def download_url(file_path: str = None, filename: str = None,
                 url: str = None, netloc: str = None):
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

def url_validator(url: str = None):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image', default=None,
        help='image to be classified')
    parser.add_argument(
        '-m', '--model', default=None,
        help='.tflite model to be executed')
    parser.add_argument(
        '-l', '--label', default=None,
        help='name of file containing labels')

    return parser.parse_args()
