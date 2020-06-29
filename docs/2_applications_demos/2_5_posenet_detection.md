---
layout: default
title: Pose Detection
parent: Applications and Demos
nav_order: 5
---

# **Pose Detection**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

In computer vision and robotics, a typical task is to identify specific objects
in an image and to determine each object's position and orientation relative to
some coordinate system. This information can then be used, for example, to allow
a robot to manipulate an object or to avoid moving into the object.
The combination of position and orientation is referred to as the pose of an
object, even though this concept is sometimes used only to describe the
orientation. Exterior orientation and translation are also used as synonyms
of pose [^1] .

## **Pose Detection**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * MobileNet as default algorithm [^3] .

More details on [eIQ™][eiq] page.

**NOTE:** This demo needs a quantized model to work properly.

### **Running Pose Detection**

#### **Using Images for Inference**

##### **Default Image**

1. Run the _Pose Detection_ demo using the following line:
```console
/opt/eiq/demos# python3 eiq_pose_detection.py
```
  * This runs inference on a default image:
  ![posenet_detection][image_eiqposedetection]

##### **Custom Image**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_pose_detection.py --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the _Pose Detection_ using the following line:
```console
/opt/eiq/demos# python3 eiq_pose_detection.py --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_pose_detection.py --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_pose_detection.py --help
```

## **References**

[^1]: https://en.wikipedia.org/wiki/Pose_(computer_vision)
[^2]: https://www.tensorflow.org/lite
[^3]: https://arxiv.org/abs/1704.04861

[image_eiqposedetection]: ../media/demos/eIQPoseDetection/image_eiqposedetection_resized_logo.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
