# Traning Keras/TensorFlow Model

1. Use Virtualenv tool to create an isolated Python environment in the PyeIQ repo:
```console
~/pyeiq/eiq/train$ virtualenv env
~/pyeiq/eiq/train$ source env/bin/activate
```

  * Install the Python requirements to train the model:
  ```console
  (env) ~/pyeiq/eiq/train$ pip3 install -r requirements-fire-detection.txt
  ```

  * Train the model passing the number of epochs [0 - 50]:
  ```console
  (env) ~/pyeiq/eiq/train$ python3 train.py --epochs=50
  ```

    This generate (_.pb_) and (_.h5_) models in *output/fire_detection.model/* folder.

  * To convert the Keras (_.h5_) model to the TensorFlow Lite model, run:
  ```console
  (env) ~/pyeiq/eiq/train$ python3 convert_h5_to_tflite.py
  ```

    This generates a (.tflite) model in *output/fire_detection.model/*.

  * Deploy this model to the board and then:
  ```console
  root@imx8qmmek:~/opt/eiq/demos/# python3 fire_detection_camera.py --model=<model.tflite path>
  ```
