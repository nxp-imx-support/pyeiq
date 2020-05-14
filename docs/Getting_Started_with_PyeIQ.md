# Getting Started with PyeIQ

Before installing PyeIQ, ensure all dependencies are installed. Most of them are
common dependencies found in any _GNU/Linux Distribution_; package names will be
different, but it shouldn't be difficult to search using whatever package management
tool that's used by your distribution.

The procedures described in this document target a GNU/Linux Distribution Ubuntu 18.04.

## Software Requirements

1. Install the following packages in the GNU/Linux system:
```console
~# apt install python3 python3-pip
```

2. Then, use _pip3_ tool to install the _virtualenv_ tool:
```console
~$ pip3 install virtualenv
```

## Building the PyeIQ Package

1. Clone the PyeIQ repository from CAF.
```console
~$ git clone https://source.codeaurora.org/external/imxsupport/pyeiq
~$ cd pyeiq/
~/pyeiq$ git checkout tag_v<latest_version>
```

2. Use _Virtualenv_ tool to create an isolated Python environment:
```console
~/pyeiq$ virtualenv env
~/pyeiq$ source env/bin/activate
```
3. Generate the PyeIQ package:
```console
(env) ~/pyeiq# python3 setup.py sdist
```
4. Copy the package to the board:
```console
(env) ~/pyeiq$ scp dist/eiq-<version>.tar.gz root@<boards_IP>:~
```

3. To deactivate the virtual environment:
```console
(env) ~/pyeiq$ deactivate
~/pyeiq$
```

## Deploy the PyeIQ Package

1. Install the PyeIQ package file in the board:
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
    The output is the PyeIQ latest version installed in the system.

## Running Applications and Demos

The demos and applications are installed in the `/opt/eiq/` folder.

1. To run the demos:

    * Choose the demo and execute it:
    ```console
    root@imx8qmmek:~# cd /opt/eiq/demos/
    root@imx8qmmek:~/opt/eiq/demos/# python3 <demo>.py
    ```
2. To run the apps:

    * Choose the app and execute it:
    ```console
    root@imx8qmmek:~# cd /opt/eiq/apps/
    root@imx8qmmek:~/opt/eiq/apps/# python3 <app>.py
    ```
3. Use help if needed:
    ```console
    root@imx8qmmek:~/opt/eiq/demos/# python3 <demo>.py --help
    root@imx8qmmek:~/opt/eiq/apps/# python3 <app>.py --help
    ```

## Running Applications and Demos from Interactive Shell

1. Open an interactive Python3 shell in the terminal:
```console
root@imx8qmmek:~# python3
```

2. Create an object using the demo or application class and run it:
```console
>>> from eiq.<framework>.<machine_learning_type> import <pyeiq_class>
>>> app = <pyeiq_class>()
>>> app.run()
```
