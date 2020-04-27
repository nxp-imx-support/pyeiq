# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.classification import eIQObjectDetection


def main():
    app = eIQObjectDetection()
    app.run()


if __name__ == '__main__':
    main()
