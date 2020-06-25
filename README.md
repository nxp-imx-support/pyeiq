# PyeIQ

![pip3][eiqpackage]
[![PyPI version](https://badge.fury.io/py/eiq.svg)](https://badge.fury.io/py/eiq)
![GitHub issues][license]
[![Gitter][gitter-image]][gitter-url]

##  A Python Framework for eIQ on i.MX Processors

PyeIQ is written on top of [eIQ™ ML Software Development Environment][eiq] and
provides a set of Python classes allowing the user to run Machine Learning
applications in a simplified and efficiently way without spending time on
cross-compilations, deployments or reading extensive guides.

### Official Releases

| **PyeIQ Version**     | **Release Date** | **i.MX Board** | **BSP Release**              | **Status** | **Notes** |
|-----------------------|------------------|----------------|------------------------------|---------------------|-----------|
| ![tag][tag_v1]        | April 29, 2020   | ![imx][boards] | ![BSP][release_5.4.3_2.0.0]  | ![build][passing]   | PoC       |
| ![tag][tag_v2]        | Planned for June | ![imx][boards] | ![BSP][release_5.4.24_2.1.0] | ![build][passing]   | Stable    |

### Installation for Users on i.MX Board

PyeIQ is hosted on [PyPI][pypirepo] repository referring to the latest tag on [CAF][pypicaf].

1. Use _pip3_ tool to install the package:

 ![PyPI](guides/media/pypieiq.gif)

2. Find the installation at **/opt/eiq** folder. Use **--help** to learn how to run the sample applications.

### Samples

| Object Classification (~3ms)   | Object Detection (~15ms)  |
|-------------------------------|--------------------------|
| ![oc](guides/media/car_classification.gif)  | ![od](guides/media/car_detection.gif) |

### Copyright and License

© 2020 NXP Semiconductors.

Free use of this software is granted under the terms of the BSD 3-Clause License.

[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
[eiqpackage]: https://img.shields.io/badge/pip3%20install-eiq-green
[pypirepo]: https://pypi.org/project/eiq/#description
[pypicaf]: https://source.codeaurora.org/external/imxsupport/pyeiq/
[license]: https://img.shields.io/badge/License-BSD%203--Clause-blue
[gitter-url]: https://gitter.im/pyeiq-imx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[gitter-image]: https://badges.gitter.im/pyeiq-imx/community.svg
[boards]: https://img.shields.io/badge/-8QM%2C%208MPlus-lightgrey
[release_5.4.3_2.0.0]: https://img.shields.io/badge/-5.4.3__2.0.0-blueviolet
[release_5.4.24_2.1.0]: https://img.shields.io/badge/-5.4.24__2.1.0-blueviolet
[tag_v1]: https://img.shields.io/badge/-v1.0.0-blue
[tag_v2]: https://img.shields.io/badge/-v2.0.0-blue
[tag_v3]: https://img.shields.io/badge/-v3.0.0-blue
[passing]: https://img.shields.io/badge/Build-passing-success
