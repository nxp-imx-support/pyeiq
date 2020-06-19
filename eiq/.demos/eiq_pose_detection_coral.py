#!/usr/bin/env python3
# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.detection.pose_detection import eIQCoralPoseNet


def main():
    app = eIQCoralPoseNet()
    app.run()


if __name__ == '__main__':
    main()
