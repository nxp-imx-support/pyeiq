# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.ssd.classification import eIQObjectDetectionSSD


def main():
    app = eIQObjectDetectionSSD()
    app.run()


if __name__ == '__main__':
    main()
