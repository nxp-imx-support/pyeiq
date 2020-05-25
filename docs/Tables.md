## Release Date

| PyeIQ Version | Release Date | Notes                |
|---------------|--------------|----------------------|
| tag_v1.0      | Apr 29, 2020 | -                    |
| tag_v2.0.0    | -            | Planned for June     |

## BSP Support

| **i.MX Board** | **BSP Release**   | **PyeIQ Version Support** | **Building Status** |
|----------------|-------------------|---------------------------|---------------------|
| 8 QM           | _5.4.3_2.0.0_     | tag_v1.0                  | **passing**         |
| 8 MPlus        | _5.4.3_2.0.0_     | tag_v1.0                  | **passing**         |

## Available Applications and Demos

| Application Name             | Framework            | i.MX Board     | BSP Release     | PyeIQ Release     | Inference Core     | Status            | Notes                   |
|------------------------------|----------------------|----------------|-----------------|-------------------|--------------------|-------------------|-------------------------|
| Object Classification        | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Object Detection SSD         | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Object Detection YOLO        | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] | Need quantizated model. |
| Object Detection DNN         | ![Framework][opencv] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Object Detection NEO DLR     | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Emotional Detection          | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Fire Classification          | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v1_v2] | ![Core][gpunpu]    | ![build][passing] |                         |
| Posenet Detection            | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Face/Eyes Detection          | ![Framework][opencv] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |
| Object Classification Switch | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v1_v2] | ![Core][cpugpunpu] | ![build][passing] |                         |
| Object Detection Switch      | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][cpugpunpu] | ![build][passing] |                         |
| Application Player           | ![Framework][tflite] | ![imx][boards] | ![BSP][release] | ![tag][tag_v2]    | ![Core][gpunpu]    | ![build][passing] |                         |


[boards]: https://img.shields.io/badge/-QM%2C%20MM%2C%20MPlus-lightgrey
[opencv]: https://img.shields.io/badge/OpenCV-4.2.0-yellow
[tflite]: https://img.shields.io/badge/TFLite-2.1.0-orange
[release]: https://img.shields.io/badge/-5.4.3__2.0.0-blueviolet
[gpunpu]: https://img.shields.io/badge/-GPU%2C%20NPU-green
[cpugpunpu]: https://img.shields.io/badge/-CPU%2C%20GPU%2C%20NPU-green
[tag_v1]: https://img.shields.io/badge/-v1.0.0-blue
[tag_v2]: https://img.shields.io/badge/-v2.0.0-blue
[tag_v1_v2]:  https://img.shields.io/badge/-v1.0.0%2C%20v2.0.0-blue
[passing]: https://img.shields.io/badge/Build-passing-success
