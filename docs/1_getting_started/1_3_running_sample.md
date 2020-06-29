---
layout: default
title: Running Examples
parent: Getting Started
nav_order: 3
---

# Running Samples
{: .no_toc }

1. TOC
{:toc}
---

The demos and applications are installed in the **/opt/eiq/** folder.

## Applications
1. Choose the app and execute it:
```console
# cd /opt/eiq/apps/
# python3 <app_name>.py
```

## Demos

1. Choose the demo and execute it:
```console
# cd /opt/eiq/demos/
# python3 <demo_name>.py
```

2. Use help if needed:
```console
# python3 <demo_name>.py --help
```

 | Argument       | Description                                                                                                                                                                                                                                                               | Example of usage                                                                                                                            |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| --download -d  | Choose from which server the models are going to download. Options: drive, github, wget. If none is specified, the application search automatically for the best server.                                                                                                  | /opt/eiq/demos# eiq_demo.py --download drive /opt/eiq/demos# eiq_demo.py -d github                                                          |
| --help -h      | Shows all available arguments for a certain demo and a brief explanation of its usage.                                                                                                                                                                                    | /opt/eiq/demos# eiq_demo.py --help /opt/eiq/demos# eiq_demo.py -h                                                                           |
| --image -i     | Use an image of your choice within the demo.                                                                                                                                                                                                                              | /opt/eiq/demos# eiq_demo.py --image /home/root/image.jpg /opt/eiq/demos# eiq_demo.py -i /home/root/image.jpg                                |
| --labels -l    | Use a labels file of your choice within the demo. Labels and models must be compatible for proper outputs.                                                                                                                                                                | /opt/eiq/demos# eiq_demo.py --labels /home/root/labels.txt /opt/eiq/demos# eiq_demo.py -l /home/root/labels.txt                             |
| --model -m     | Use a model file of your choice within the demo. Labels and models must be compatible for proper outputs.                                                                                                                                                                 | /opt/eiq/demos# eiq_demo.py --model /home/root/model.tflite /opt/eiq/demos# eiq_demo.py -m /home/root/model.tflite                          |
| --res -r       | Choose the resolution of your video capture device. Options: full_hd (1920x1080), hd (1280x720), vga (640x480). If none is specified, it uses hd as default. If your video device doesn't support the chosen resolution, it automatically selects the best one available. | /opt/eiq/demos# eiq_demo.py --res full_hd /opt/eiq/demos# eiq_demo.py -r vga                                                                |
| --video_fwk -f | Choose which video framework is used to display the video. Options: opencv, v4l2, gstreamer (need improvements). If none is specified, it uses v4l2 as default.                                                                                                           | /opt/eiq/demos# eiq_demo.py --video_fwk opencv /opt/eiq/demos# eiq_demo.py -f v4l2                                                          |
| --video_src -v | It makes the demo run inference on a video instead of an image. You can simply use the parameter "True" for it to run, specify your video capture device or even a video file. Options: True, /dev/video<x>, path_to_your_video_file.                                     | /opt/eiq/demos# eiq_demo.py --video_src /dev/video3 /opt/eiq/demos# eiq_demo.py -v True /opt/eiq/demos# eiq_demo.py -v /home/root/video.mp4 |


## Available Applications and Demos

See the **Applications and Demos** sections to check the available ones.
