#!/usr/bin/env python3
# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.classification.classification import eIQObjectsClassification


def main():
    app = eIQObjectsClassification()
    app.run()


if __name__ == '__main__':
    main()
