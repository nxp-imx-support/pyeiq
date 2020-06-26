---
layout: default
title: Object Classification
parent: Applications and Demos
nav_order: 1
---

# Object Classification
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

Object Classification is the problem of identifying to which of a set of
categories a new observation belongs, on the basis of a training set of data
containing observations whose category membership is known. Examples are
assigning a given email to the "spam" or "non-spam" class, and assigning a
diagnosis to a given patient based on observed characteristics of the patient [^1] .

## **Object Classification MobileNet**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * MobileNet as default algorithm [^3] .

More details on [eIQ™][eiq] page.

### **Running Object Classification**

#### **Using Images for Inference**

##### **Default Image**

1. Run the Object Classification demo using the following line:
```console
/opt/eiq/demos# python3 eiq_objects_classification_tflite.py
```
  * This runs inference on a default image:
  ![classification][image_eIQObjectClassification]

##### **Custom Image**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_objects_classification_tflite.py --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the Object Classification using the following line:
```console
/opt/eiq/demos# python3 eiq_objects_classification_tflite.py --video_src=/path_to_the_video
```
  * This runs inference on a video file:
  ![classification_video][video_eIQObjectClassification]

#### **Video Camera or Webcam**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_objects_classification_tflite.py --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_objects_classification_tflite.py --help
```

## **References**

[^1]: https://en.wikipedia.org/wiki/Statistical_classification
[^2]: https://www.tensorflow.org/lite
[^3]: https://arxiv.org/abs/1704.04861

[image_eIQObjectClassification]: ../media/demos/eIQObjectClassification/image_eIQObjectClassification_resized_logo.gif
[video_eIQObjectClassification]: ../media/demos/eIQObjectClassification/video_eIQObjectClassification.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
