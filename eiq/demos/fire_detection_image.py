# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.classification import eIQFireDetection


def main():
    app = eIQFireDetection()
    app.run()


if __name__ == '__main__':
    main()
