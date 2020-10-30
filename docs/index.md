---
layout: default
title: Home
nav_order: 1
---

<p align="center">
  <img src="media/logo/color/pyeiq_alpha_logo.png" height="191" width="176">
</p>

<h1 align="center">
<b>A Python Framework for eIQ on i.MX Processors</b>
</h1>

[PyeIQ][caf] is written on top of [eIQ™ ML Software Development Environment][eiq]
and provides a set of Python classes allowing the user to run Machine Learning
applications in a simplified and efficiently way without spending time on
cross-compilations, deployments or reading extensive guides.


* **Take as a disclaimer that PyeIQ should not be considered production-ready**.
* For further questions, please post a comment on [eIQ™ community][page].

![open_source.png](media/OSI_Standard_Logo_100X130.png)                                           |
:--------------------------------------------------------------------------------------:  |
**Free and Open Source**: Framework under BSD-3-Clause fully extensible and customizable. |
**Ready-to-use**: Cutting-edge ML samples demonstrating full power of the of framework.   |


### **Samples**

| **Object Classification (~3ms)**              | **Object Detection (~15ms)**             |
|-----------------------------------------------|------------------------------------------|
| ![oc_1][video_eIQObjectClassification_room]   | ![od_1][video_eIQObjectDetection_room]   |
| ![oc_2][video_eIQObjectClassification_street] | ![od_2][video_eIQObjectDetection_street] |


### Official Releases

| BSP Release                  | PyeIQ Release       | PyeIQ Updates    | Board          | Date      | Status             | Notes   |
|------------------------------|---------------------|------------------|----------------|-----------|--------------------|---------|
| ![BSP][release_5.4.3_2.0.0]  | ![tag][tag_v100]    |                  | ![imx][boards] | Apr, 2020 | ![Build][passing]  | PoC     |
|                              |                     | ![tag][tag_v101] | ![imx][boards] | May, 2020 | ![Build][passing]  |         |
| ![BSP][release_5.4.24_2.1.0] | ![tag][tag_v200]    |                  | ![imx][boards] | Jun, 2020 | ![Build][passing]  | Stable  |
|                              |                     | ![tag][tag_v201] | ![imx][boards] | Jun, 2020 | ![Build][passing]  |         |
|                              |                     | ![tag][tag_v210] | ![imx][boards] | Aug, 2020 | ![Build][passing]  |         |
| ![BSP][release_5.4.47_2.2.0] |                     | ![tag][tag_v220] | ![imx][boards] | Nov, 2020 | ![Build][passing]  |         |

![blue][tag_blue]
![yellow][tag_yellow]
![red][tag_red]

### Major Changes

**2.0.0**
- General major changes on project structure.
- Split project into engine, modules, helpers, utils and apps.
- Add base class to use on all demos avoiding repeated code.
- Support for more demos and applications including Arm NN.
- Support for building using Docker.
- Support for download data from multiple servers.
- Support for searching devices and build pipelines.
- Support for appsink/appsrc for QM (not working on MPlus).
- Support for camera and H.264 video.
- Support for Full HD, HD and VGA resolutions.
- Support video and image for all demos.
- Add display info in the frame, such as: FPS, model and inference time.
- Add manager tool to launch demos and applications.
- Add document page for PyeIQ project.

**1.0.0**
- Support demos based on TensorFlow Lite (2.1.0) and image classification.      
- Support inference running on GPU/NPU and CPU.
- Support file and camera as input data.
- Support SSD (Single Shot Detection).
- Support downloads on the fly (models, labels, dataset, etc).
- Support old eIQ demos from eiq_sample_apps CAF repository.
- Support model training for host PC.
- Support UI for switching inference between GPU/NPU/CPU on TensorFlow Lite.

### Copyright and License

Copyright 2020 NXP Semiconductors. Free use of this software is granted under
the terms of the BSD 3-Clause License.
See [LICENSE](https://source.codeaurora.org/external/imxsupport/pyeiq/tree/LICENSE.md?h=v2.0.0)
for details.

[video_eIQObjectDetection_room]: media/demos/eIQObjectDetection/video_eIQObjectDetection_room.gif
[video_eIQObjectClassification_room]: media/demos/eIQObjectClassification/video_eIQObjectClassification_room.gif

[video_eIQObjectDetection_street]: media/demos/eIQObjectDetection/video_eIQObjectDetection_street.gif
[video_eIQObjectClassification_street]: media/demos/eIQObjectClassification/video_eIQObjectClassification_street.gif

[page]: https://community.nxp.com/t5/eIQ-Machine-Learning-Software/bd-p/eiq

[caf]: https://source.codeaurora.org/external/imxsupport/pyeiq/
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
[boards]: https://img.shields.io/badge/-8QM%2C%208MPlus-lightgrey
[passing]: https://img.shields.io/badge/Build-passing-success

[release_5.4.3_2.0.0]: https://img.shields.io/badge/-5.4.3__2.0.0-blueviolet
[release_5.4.24_2.1.0]: https://img.shields.io/badge/-5.4.24__2.1.0-blueviolet
[release_5.4.47_2.2.0]: https://img.shields.io/badge/-5.4.47__2.2.0-blueviolet

[tag_blue]: https://img.shields.io/badge/-new-blue
[tag_yellow]: https://img.shields.io/badge/-features-yellow
[tag_red]: https://img.shields.io/badge/-bug%20fixes-red

[tag_v100]: https://img.shields.io/badge/-v1.0.0-blue
[tag_v101]: https://img.shields.io/badge/-v1.0.1-red
[tag_v110]: https://img.shields.io/badge/-v1.1.0-red

[tag_v200]: https://img.shields.io/badge/-v2.0.0-blue
[tag_v201]: https://img.shields.io/badge/-v2.0.1-red
[tag_v210]: https://img.shields.io/badge/-v2.1.0-yellow
[tag_v220]: https://img.shields.io/badge/-v2.2.0-red
