---
layout: default
title: Switch Detection Video
parent: Applications and Demos
nav_order: 7
---

# **Switch Detection Video**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

This application offers a graphical interface for users to run an object
detection demo using either CPU or GPU/NPU to perform inference on a video file.

## **Switch Detection Video**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This application uses:

 * TensorFlow Lite as an inference engine [^1] ;
 * MobileNet as default algorithm [^2] .

More details on [eIQ™][eiq] page.

### **Running Switch Detection Video**

1. Run the _Switch Detection Video_ demo using the following line:
```console
# pyeiq --run switch_video
```
2. Type on **CPU** or **GPU**/**NPU** in the terminal to switch between cores.

  * This result is below:
  ![cpu][switch_detection]

[^1]: https://www.tensorflow.org/lite
[^2]: https://arxiv.org/abs/1704.04861

[switch_detection]: ../media/apps/eIQVideoSwitchCore/switch_detection_resized_logo.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
