# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.detection.object_detection_sdd import eIQObjectDetectionCamera


def main():
    app = eIQObjectDetectionCamera()
    app.run()


if __name__ == '__main__':
    main()
