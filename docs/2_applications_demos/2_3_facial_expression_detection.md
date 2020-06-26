---
layout: default
title: Facial Detection
parent: Applications and Demos
nav_order: 3
---

# **Facial Detection**
{: .no_toc }

1. TOC
{:toc}
---

## **Overview**

Emotion recognition is the process of identifying human emotion. People vary
widely in their accuracy at recognizing the emotions of others. Use of technology
to help people with emotion recognition is a relatively nascent research area.
Generally, the technology works best if it uses multiple modalities in context.
To date, the most work has been conducted on automating the recognition of facial
expressions from video, spoken expressions from audio, written expressions from
text, and physiology as measured by wearables [^1] .


## **Facial Expression Detection**

### **Inference Engine and Algorithm**

![tfliteframework][tflite]

This demo uses:

 * TensorFlow Lite as an inference engine [^2] ;
 * MobileNet as default algorithm [^3] .

More details on [eIQ™][eiq] page.

### **Running Facial Expression Detection**

#### **Using Images for Inference**

##### **Default Image**

1. Run the Object Detection demo using the following line:
```console
/opt/eiq/demos# python3 eiq_facial_expression_detection.py
```
  * This runs inference on a default image:
  ![facial_detection][image_eiqfacialexpressiondetection]

##### **Custom Image**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_facial_expression_detection.py --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the Object Detection using the following line:
```console
/opt/eiq/demos# python3 eiq_facial_expression_detection.py --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_facial_expression_detection.py --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_facial_expression_detection.py --help
```

## **Facial and Eyes Detection**

### **Inference Engine and Algorithm**

![opencvframework][opencv]

This demo uses:

 * OpenCV as engine [^4] ;
 * Haar Cascades as default algorithm [^5] .

More details on [eIQ™][eiq] page.

### **Running Facial Expression Detection**

#### **Using Images for Inference**

##### **Default Image**

1. Run the Object Detection demo using the following line:
```console
/opt/eiq/demos# python3 eiq_face_and_eyes_detection.py
```
  * This runs inference on a default image:
  ![face_detection][image_eiqfaceandeyesdetection]

##### **Custom Image**

1. Pass any image as an argument:
```console
/opt/eiq/demos# python3 eiq_face_and_eyes_detection.py --image=/path_to_the_image
```

#### **Using Video Source for Inference**

##### **Video File**

1. Run the Object Detection using the following line:
```console
/opt/eiq/demos# python3 eiq_face_and_eyes_detection.py --video_src=/path_to_the_video
```

##### **Video Camera or Webcam**

1. Specify the camera device:
```console
/opt/eiq/demos# python3 eiq_face_and_eyes_detection.py --video_src=/dev/video<index>
```

### **Extra Parameters**

1. Use **--help** argument to check all the available configurations:
```console
/opt/eiq/demos# python3 eiq_face_and_eyes_detection.py --help
```

## **References**

[^1]: https://en.wikipedia.org/wiki/Emotion_recognition
[^2]: https://www.tensorflow.org/lite
[^3]: https://arxiv.org/abs/1704.04861
[^4]: https://github.com/opencv/opencv
[^5]: https://github.com/opencv/opencv/tree/master/data/haarcascades

[image_eiqfacialexpressiondetection]: ../media/demos/eIQFacialExpressionDetection/image_eiqfacialexpressiondetection_resized_logo.gif

[image_eiqfaceandeyesdetection]: ../media/demos/eIQFaceAndEyesDetection/image_eiqfaceandeyesdetection_resized_logo.gif

[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[opencv]: https://img.shields.io/badge/OpenCV-4.2.0-yellow
[eiq]: https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ
