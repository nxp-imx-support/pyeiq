# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.classification import eIQObjectDetectionGStreamer


def main():
    app = eIQObjectDetectionGStreamer()
    app.run()


if __name__ == '__main__':
    main()
