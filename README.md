# Welcome to PyeIQ

PyeIQ provide high level classes to allow the user execute eIQ applications and demos.


| i.MX Board | BSP Release | Building Status |
|------------|-------------|-----------------|
| 8 QM       | 5.4         | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build") |
| 8 MPlus    | 5.4         | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build") |
| 8 M Mini   | 5.4         |         -        |

## Getting Started with PyeIQ

Before installing PyeIQ, ensure all dependencies are installed. Most of them are
common dependencies found in any _GNU/Linux Distribution_; package names will be
different, but it shouldn't be difficult to search using whatever package management
tool that's used by your distribution.

The procedures described in this document target a GNU/Linux Distribution Ubuntu 18.04.

### Software Requirements

1. Install the following packages in the GNU/Linux system:
```console
~# apt install python3 python3-pip
```

2. Then, use _pip3_ tool to install the [_Virtualenv_](https://virtualenv.pypa.io/en/latest/) tool:
```console
~$ pip3 install virtualenv
```

### Building the PyeIQ Package

1. Clone the PyeIQ repository from CAF.

2. Use _Virtualenv_ tool to create an isolated Python environment:
```console
~/pyeiq$ virtualenv env
~/pyeiq$ source env/bin/activate
```
   * Generate the PyeIQ package:
   ```console
   (env) ~/pyeiq# python3 setup.py sdist bdist_wheel
   ```
  * Copy the package to the board:
  ```console
  (env) ~/pyeiq$ scp dist/eiq-<version>.tar.gz root@<boards_IP>:~
  ```

3. To deactivate the virtual environment:
```console
(env) ~/pyeiq$ deactivate
~/pyeiq$
```

### Deploy the PyeIQ Package

1. Install the PyeIQ Wheel file in the board:
```console
root@imx8qmmek:~# pip3 install eiq-<version>.tar.gz
```

2. Check the installation:
    * Start an interactive shell mode with Python3:
    ```console
    root@imx8qmmek:~# python3
    ```

    * Check the PyeIQ latest version:
    ```console
    >>> import eiq
    >>> eiq.__version__
    ```

    * The output is the PyeIQ latest version installed in the system.

## Running the Demos

All the demos are installed in the `/opt/eiq/demos` folder. Follow a list of the
available demos in PyeIQ:

|  Demo/App Name                    |  Demo/App Type   | i.MX Board | BSP Release | BSP Framework           | Inference | Status |  Notes                                      |
|-----------------------------------|------------------|------------|-------------|-------------------------|-----------|--------|---------------------------------------------|
| Label Image                       | File Based       | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | -
| Label Image Switch                | File Based       | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | -
| Object Detection                  | SSD/Camera Based | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | Works with low accuracy. Need better model. |
| Object Detection OpenCV           | SSD/Camera Based | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | Higher accuracy than above one.             |
| Object Detection Native GStreamer | SSD/Camera Based | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | -       | Fixing undetermined GStreamer hangs.        |
| Object Detection Yolov3           | SSD/File Based   | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | -       | Pending issues.                             |
| Object Detection Yolov3           | SSD/Camera Based | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | -       | Pending issues.                             |
| Fire Detection                    | File Based       | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | -                                            |
| Fire Detection                    | Camera Based     | QM, MPlus  | _5.4_       | TensorFlow Lite _2.1.0_ | GPU, NPU  | ![build](https://img.shields.io/travis/asciidoctor/jekyll-asciidoc/master.svg "Build")       | -                                            |
| Fire Detection                    | Camera Based     | -           | _5.4_       | PyArmNN _19.08_         | -          | -       | Requires _19.11_                            |
| Coral Posenet                     |  Camera Based    | -           | -            | -                        | -          | -       | Ongoing                                     |
| NEO DLR                           | Camera Based     | -           | -            | -                        | -          | -       | Ongoing                                     |

1. To run the demos:
    * Choose the demo and execute it:
    ```bash
    root@imx8qmmek:~# cd /opt/eiq/demos/
    root@imx8qmmek:~/opt/eiq/demos/# python3 <demo>.py
    ```
    * Use help if needed:
    ```bash
    root@imx8qmmek:~/opt/eiq/demos/# python3 <demo>.py --help
   ```

## Copyright and License

© 2020 NXP Semiconductors.

Free use of this software is granted under the terms of the BSD 3-Clause License.
