# Copyright 2020 NXP Semiconductors
# SPDX-License-Identifier: BSD-3-Clause

from eiq.modules.classification.classification_images import eIQFireDetection


def main():
    app = eIQFireDetection()
    app.run()


if __name__ == '__main__':
    main()
