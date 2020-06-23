#!/usr/bin/env python3
# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.detection.pose_detection import eIQPoseDetection


def main():
    app = eIQPoseDetection()
    app.run()


if __name__ == '__main__':
    main()
