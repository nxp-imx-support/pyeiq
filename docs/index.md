---
layout: default
title: Home
nav_order: 1
---

<p align="center">
  <img src="media/pyeiq.png" height="191" width="176">
</p>

<h1 align="center">
<b>A Python Framework for eIQ on i.MX Processors</b>
</h1>

[PyeIQ][caf] is written on top of [eIQ™ ML Software Development Environment][eiq]. Provides
a set of Python classes allowing the user to run Machine Learning applications in
a simplified and efficiently way without spending time on cross-compilations,
deployments or reading extensive guides.


![ready_to_use.png](media/ready_to_use_small.png)                                       | ![open_source.png](media/open_source_small.png)
:--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------:
**Ready-to-use**: Cutting-edge ML samples demonstrating full power of the of framework.  | **Free and Open Source**: Framework under BSD-3-Clause fully extensible and customizable.


### **Samples**

| **Object Classification (~3ms)**       | **Object Detection (~15ms)**     |
|----------------------------------------|----------------------------------|
| ![oc](media/car_classification.gif)  | ![od](media/car_detection.gif) |


### **Official Releases**

| **PyeIQ Version**     | **Release Date** | **i.MX Board** | **BSP Release**        | **Status**                | **Notes** |
|-----------------------|------------------|----------------|------------------------|---------------------------|-----------|
| ![tag][tag_v1]        | Apr 29, 2020     | ![imx][boards] | ![BSP][release_5.4.3]  | ![Build][workflow-build]  | PoC       |
| ![tag][tag_v2]        | Jun 30, 2020     | ![imx][boards] | ![BSP][release_5.4.3]  | ![Build][workflow-build]  | Stable    |


[caf]: https://source.codeaurora.org/external/imxsupport/pyeiq/
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
[workflow-build]: https://github.com/diegohdorta/pyeiq/workflows/Build/badge.svg
[boards]: https://img.shields.io/badge/-8QM%2C%208MPlus-lightgrey
[release_5.4.3]: https://img.shields.io/badge/-5.4.3__2.0.0-blueviolet
[release_5.4.24]: https://img.shields.io/badge/-5.4.24__2.1.0-blueviolet
[tag_v1]: https://img.shields.io/badge/-v1.0.0-blue
[tag_v2]: https://img.shields.io/badge/-v2.0.0-blue
[passing]: https://img.shields.io/badge/Build-passing-success
