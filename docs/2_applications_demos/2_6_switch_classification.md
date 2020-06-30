---
layout: default
title: Switch Classification Image
parent: Applications and Demos
nav_order: 6
---

# **Switch Classification Image**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

This application offers a graphical interface for users to run an object
classification demo using either CPU or GPU/NPU to perform inference on a list
of available images.

## **Switch Classification Image**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This application uses:

 * TensorFlow Lite as an inference engine [^1] ;
 * MobileNet as default algorithm [^2] .

More details on [eIQ™][eiq] page.

### **Running Switch Classification Image**

1. Run the _Switch Classification Image_ demo using the following line:
```console
# pyeiq --run switch_image
```
2. Choose an image, then click on **CPU** or **GPU**/**NPU** button:

  * **CPU**:
  ![cpu][cpu_switch_classification]

  * **NPU**:
  ![npu][npu_switch_classification]

## **References**

[^1]: https://www.tensorflow.org/lite
[^2]: https://arxiv.org/abs/1704.04861

[cpu_switch_classification]: ../media/apps/eIQSwitchLabelImage/switch_classification_cpu_resized_logo.gif
[npu_switch_classification]: ../media/apps/eIQSwitchLabelImage/switch_classification_npu_resized_logo.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
