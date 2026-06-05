# YOLOv11 多类别车辆检测

> 基于 Ultralytics YOLOv11 实现的多类别车辆检测项目，涵盖环境配置、数据集准备、模型训练与视频推理全流程。

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.60-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 项目简介

本项目在**自定义车辆数据集**（12 个细分类别，如轿车、巴士、卡车等）上训练了 YOLOv11n 检测模型，实现了对交通场景中多种车辆的准确检测。完整记录了在 Windows 11 + WSL2 + NVIDIA RTX 5060 环境下的配置过程、训练调优技巧以及常见问题解决方案，方便他人复现。

- **模型**：YOLOv11n（nano）
- **任务**：多类别车辆检测（bus, car, truck 等 12 类）
- **硬件**：NVIDIA RTX 5060 Laptop GPU (8GB)
- **推理速度**：约 95 FPS (352×640)

## 📊 数据集

数据集来源：RoboFlow 上的 `Venom.v8i.yolov11` 数据集（已按 YOLO 格式组织）。

- 图片数量：训练集 ？张，验证集 963 张，测试集 ？张（共 13450 个实例）
- 类别（12 类）：
  ```text
  bus-l-, bus-s-, car, extra large truck, large bus, medium truck,
  small bus, small truck, truck-l-, truck-m-, truck-s-, truck-xl-
