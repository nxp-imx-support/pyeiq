# A Python Framework for eIQ on i.MX Processors

[![Gitter][gitter-image]][gitter-url]
![GitHub issues][license]

PyeIQ provides a simplified way to run ML applications, which avoids spending time on cross-compilations, flashings, deployments or reading extensive guides.

## BSP Support

| **PyeIQ Tag Version** | **Release Date** | **i.MX Board** | **BSP Release** | **Building Status** | **Notes** |
|-----------------------|------------------|----------------|-----------------|---------------------|-----------|
| ![tag][tag_v1]        | April 29, 2020   | ![imx][boards] | ![BSP][release] | ![build][passing]   | PoC       |
| ![tag][tag_v2]        | Planned for June | -              | -               | -                   | -         |
| ![tag][tag_v2]        | Planned for Sept | -              | -               | -                   | -         |

## Installing

PyeIQ is hosted on [PyPI](https://pypi.org/project/eiq/#description) repository referring to the latest tag on [CAF](https://source.codeaurora.org/external/imxsupport/pyeiq/).

1. Use _pip3_ tool to install the package:

 ![PyPI](docs/media/pypieiq.gif)

## Samples

| Object Classification (3ms)   | Object Detection (15ms)  |
|-------------------------------|--------------------------|
| ![oc](docs/media/car_classification.gif)  | ![od](docs/media/car_detection.gif) |

## Copyright and License

© 2020 NXP Semiconductors.

Free use of this software is granted under the terms of the BSD 3-Clause License.

[license]: https://img.shields.io/badge/License-BSD%203--Clause-blue
[gitter-url]: https://gitter.im/pyeiq-imx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[gitter-image]: https://badges.gitter.im/pyeiq-imx/community.svg



[boards]: https://img.shields.io/badge/-8QM%2C%208MPlus-lightgrey
[release]: https://img.shields.io/badge/-5.4.3__2.0.0-blueviolet
[tag_v1]: https://img.shields.io/badge/-v1.0.0-blue
[tag_v2]: https://img.shields.io/badge/-v2.0.0-blue
[tag_v3]: https://img.shields.io/badge/-v3.0.0-blue
[passing]: https://img.shields.io/badge/Build-passing-success
