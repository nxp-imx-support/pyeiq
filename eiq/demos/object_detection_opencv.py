# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.detection.object_detection_ssd import eIQObjectDetectionOpenCV


def main():
    app = eIQObjectDetectionOpenCV()
    app.run()


if __name__ == '__main__':
    main()
