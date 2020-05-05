# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.neodlr.ssd.classification import eIQObjectDetectionDLR


def main():
    app = eIQObjectDetectionDLR()
    app.run()


if __name__ == '__main__':
    main()
