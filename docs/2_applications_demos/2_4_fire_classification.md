---
layout: default
title: Fire Classification
parent: Applications and Demos
nav_order: 4
---

# **Fire Classification**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

Fire classification and detection has recently played a crucial role in reducing
fire losses by alarming users early through early fire detection. Image fire
detection is based on an algorithmic analysis of images.
However, there is a lower accuracy, delayed detection, and a large amount of
computation in common detection algorithms, including manually and machine
automatically extracting image features [^1] .


## **Fire Classification**

### **Inference Engine and Algorithm**

![tfliteframework][tflite] ![armnnframework][armnn]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * ArmNN as an inference engine [^3] ;
 * CNN as default algorithm [^4] .

More details on [eIQ™][eiq] page.

### **Running Fire Classification**

#### **Using Images for Inference**

##### **Default Image**

* **TensorFlow Lite**

1. Run the _Fire Classification_ demo using the following line:
```console
/opt/eiq/demos# python3 eiq_fire_classification_tflite.py
```

* **Arm NN**

1. Run the _Fire Classification_ demo using the following line:
```console
/opt/eiq/demos# python3 eiq_fire_classification_armnn.py
```

* This runs inference on a default image:
![facial_detection][image_eiqfireclassification]

##### **Custom Image**

* **TensorFlow Lite**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_fire_classification_tflite.py --image=/path_to_the_image
```

* **Arm NN**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_fire_classification_armnn.py --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

* **TensorFlow Lite**

1. Run the _Fire Classification_ using the following line:
```console
/opt/eiq/demos# python3 eiq_fire_classification_tflite.py --video_src=/path_to_the_video
```

* **Arm NN**

1. Run the _Fire Classification_ using the following line:
```console
/opt/eiq/demos# python3 eiq_fire_classification_armnn.py --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

* **TensorFlow Lite**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_fire_classification_tflite.py --video_src=/dev/video<index>
```

* **Arm NN**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_fire_classification_armnn.py --video_src=/dev/video<index>
```

### **Extra Parameters**

* **TensorFlow Lite**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_fire_classification_tflite.py --help
```

* **Arm NN**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_fire_classification_armnn.py --help
```

## **References**

[^1]: https://doi.org/10.1016/j.csite.2020.100625
[^2]: https://www.tensorflow.org/lite
[^3]: https://github.com/ARM-software/armnn
[^4]: https://en.wikipedia.org/wiki/Convolutional_neural_network


[image_eiqfireclassification]: ../media/demos/eIQFireClassification/image_eiqfireclassification_resized_logo.gif


[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[armnn]: https://img.shields.io/badge/ArmNN-19.08-blue
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
