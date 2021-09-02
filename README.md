该mmseg为处理ocr版面分割所用的主要框架，对其进行一些修改：
环境配置 :  cuda 10.0，gcc 4.9.2， torch 1.3， torchvision 0.4.1 
mmcv 0.3.1   mmcv-full 1.3.7

* 分层多尺度
  文件目录303行见注释
  Mmseg/mmseg/models/segmentors/encoder_decoder.py
  <div align="center">
    <img src="readme_png/multiscale-diflayer.png" width="600"/>
  </div>
  <br />

* mmseg分类与分割并行
  在voc格式数据集中指定位置，需要有cls-label.json。包含了每张图对应的标签，通用文档-0，ppt-1，格式如下。
  <div align="center">
    <img src="readme_png/cls0.png" width="600"/>
  </div>
  <br />
  <div align="center">
    <img src="readme_png/cls1.png" width="600"/>
  </div>
  <br />
  在Mmseg/mmseg/datasets/custom.py中对cls-label.json 进行读取并保存，用于计算指标，保存方式如下
  <div align="center">
    <img src="readme_png/cls2.png" width="600"/>
  </div>
  <br />
  训练时需要用到cls-label，(在200行)准备训练数据时将cls-label加入到img-metas中去
  <div align="center">
    <img src="readme_png/cls3.png" width="600"/>
  </div>
  <br />
  测试时需要对分类结果进行accuracy指标输出。在318行编写了cls-acc函数用于计算精度
  <div align="center">
    <img src="readme_png/cls4.png" width="600"/>
  </div>
  <br />
  在/Mmseg/mmseg/models/segmentors/encoder_decoder.py中，设置cls-head和lossfn用于计算分类的out和loss，由于分类与分割公用backbone，所以decode-head的输入通道数即为cls-head中全连接层的输入维度。
  <div align="center">
    <img src="readme_png/cls5.png" width="600"/>
  </div>
  <br />
  在160行进行分类loss，acc的计算并更新到total-loss中用于整体梯度更新
  <div align="center">
    <img src="readme_png/cls6.png" width="600"/>
  </div>
  <br />
  在测试过程中增加返回值cls-out，同时更改所用到的所有测试流程函数
  <div align="center">
    <img src="readme_png/cls7.png" width="600"/>
  </div>
  <br />
  Mmseg/mmseg/apis/test.py中保存整体cls的预测label，最后回到datasets进行计算指标
  <div align="center">
    <img src="readme_png/cls9.png" width="600"/>
  </div>
  <br />

* 已尝试内容
  模型：
    deeplabv3+(resnet)
    deeplabv3+(resnet+dcn)
    deeplabv3+(resnest)
    vit
    swin
    setr
    hrnet
    对比上述模型v3+（resnet）在ocr版面分割任务中前向速度最快，指标也最高
  方法：
    分层多尺度：见上面4（有效）
    标签腐蚀：固定中心坐标不变，将每一个目标的mask覆盖区域进行收缩处理，原图不进行处理。（有效）

    <div align="center">
      <img src="readme_png/erode.png" width="600"/>
    </div>
    <br />

    booststrap：标签腐蚀的升级版，将mask进行随机大幅度裁剪而非小尺度收缩，之后对原图进行背景均化（有效）
    * booststrap前

    <div align="center">
      <img src="readme_png/boost0.png" width="600"/>
    </div>
    <br />

    * booststrap后

    <div align="center">
      <img src="readme_png/boost1.png" width="600"/>
    </div>
    <br />

* trace流程
  在Mmseg/trace_tools下
    1.trace_cpu.py  ===== 将pth模型转换为pt模型
      --model_path：  pth模型地址
      --dst-trace-path：想要保存的pt模型文件		
      --config-path:  pth模型对应的config文件
      -- img-path： 任意图片
    2.trace-cpu-valid.py ===== 验证pth模型与pt模型是否一致
      --model-path： pt模型地址
      --pytorch-model-path： pth模型地址
      --config-path：pth模型对应的config
      -- img-path：任意图片
      --output-path：可视化输出目录
      --cls-dict： 分类标签对应内容


* 官方Mmsegmentation指引
<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br />

[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

Documentation: https://mmsegmentation.readthedocs.io/

English | [简体中文](README_zh-CN.md)

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.3+**.

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.14.1 was released in 06/16/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet/README.md)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest/README.md)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2/README.md)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3/README.md)
- [x] [Vision Transformer (ICLR'2021)]

Supported methods:

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [SETR (CVPR'2021)](configs/setr)

## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [train.md](docs/train.md) and [inference.md](docs/inference.md) for the basic usage of MMSegmentation.
There are also tutorials for [customizing dataset](docs/tutorials/customize_datasets.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing modules](docs/tutorials/customize_models.md), and [customizing runtime](docs/tutorials/customize_runtime.md).
We also provide many [training tricks](docs/tutorials/training_tricks.md).

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) on Colab.

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): A powerful toolkit for generative models.
