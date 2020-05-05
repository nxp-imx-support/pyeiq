# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.ssd.classification import eIQObjectDetectionCamera


def main():
    app = eIQObjectDetectionCamera()
    app.run()


if __name__ == '__main__':
    main()
