# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.tflite.classification import eIQLabelImage


def main():
    app = eIQLabelImage()
    app.run()


if __name__ == '__main__':
    main()
