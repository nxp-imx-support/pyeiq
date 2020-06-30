#!/usr/bin/env python3
# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.classification.classification_tflite import eIQObjectClassificationTFLite


def main():
    app = eIQObjectClassificationTFLite()
    app.run()


if __name__ == '__main__':
    main()
