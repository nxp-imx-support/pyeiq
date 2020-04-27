# TFlite Runtime Examples

## Building Process

Refers to the first README file.

## Running

1. Open the Python3 terminal:
```console
root@imx8qmmek:~# python3
```

  * Label Image
  ```console
  >>> from eiq.tflite.classification import eIQLabelImage
  >>> app = eIQLabelImage()
  >>> app.run()
  ```

  * Object Detection Camera
  ```console
  >>> from eiq.tflite.classification import eIQObjectDetection
  >>> app = eIQObjectDetection()
  >>> app.run()
  ```

  * Object Detection OpenCV
  ```console
  >>> from eiq.tflite.classification import eIQObjectDetectionOpenCV
  >>> app = eIQObjectDetectionOpenCV()
  >>> app.run()
  ```

  * Object Detection GStreamer
  ```console
  >>> from eiq.tflite.classification import eIQObjectDetectionGStreamer
  >>> app = eIQObjectDetectionGStreamer()
  >>> app.run()
  ```

  * Fire Detection Image
  ```console
  >>> from eiq.tflite.classification import eIQFireDetection
  >>> app = eIQFireDetection()
  >>> app.run()
  ```

  * Fire Detection Camera
  ```console
  >>> from eiq.tflite.classification import eIQFireDetectionCamera
  >>> app = eIQFireDetectionCamera()
  >>> app.run()
  ```
