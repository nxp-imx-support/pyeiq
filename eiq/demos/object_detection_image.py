# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.opencv.classification import eIQSingleShotObjectDetection


def main():
    app = eIQSingleShotObjectDetection()
    app.run()


if __name__ == '__main__':
    main()
