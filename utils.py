import os
import pathlib
import shutil
import sys

def copy(target_dir, src_dir):
    init = "__init__.py"

    if not os.path.exists(target_dir):
        try:
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            sys.exit("os.mkdir() function has failed: %s" % target_dir)

    for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)

            if os.path.isdir(file_path):
                copy(os.path.join(target_dir, file), file_path)
            else:
                if file != init:
                    shutil.copy(file_path, target_dir)