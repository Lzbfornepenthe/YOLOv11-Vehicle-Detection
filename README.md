# YOLOv11 多类别车辆检测：完整实践记录

> 本文档详细记录了基于 YOLOv11 进行多类别车辆检测的完整流程，包括 Windows + WSL2 环境配置、数据集准备、模型训练、调优、推理部署以及所有常见问题的解决方案。目标是让读者能够完整复现本项目。

**项目状态**：✅ 已完成 80 轮训练，模型在验证集上 mAP50 达 0.441（12 类），汽车类 mAP50 达 0.817。推理速度约 95 FPS (RTX 5060)。

## 目录
1. [项目背景与目标](#项目背景与目标)
2. [硬件与软件环境](#硬件与软件环境)
3. [环境配置全过程](#环境配置全过程)
4. [数据集准备](#数据集准备)
5. [模型训练](#模型训练)
6. [模型评估](#模型评估)
7. [视频推理部署](#视频推理部署)
8. [项目复现指南](#项目复现指南)
9. [项目总结与改进方向](#项目总结与改进方向)
10. [致谢与许可](#致谢与许可)

## 项目背景与目标

在自动驾驶、智能交通监控等领域，实时准确的车辆检测至关重要。YOLO 系列模型以其端到端、速度快、精度高的特点成为主流选择。本项目采用最新的 **YOLOv11** 模型，在包含 12 类车辆（轿车、巴士、卡车等）的交通数据集上进行训练，实现多类别车辆检测，并验证其在实际视频中的推理性能。

**本项目完整记录了从零开始的全部步骤**，包括：
- Windows 11 + WSL2 下的 GPU 环境配置（解决网络、路径、驱动等问题）
- RoboFlow 数据集的下载、组织与 YOLO 格式转换
- 超参数选择与 80 轮训练过程
- 模型评估与视频推理
- 遇到的典型错误及解决方案

## 硬件与软件环境

| 组件               | 配置                                                       |
| ------------------ | ---------------------------------------------------------- |
| 操作系统           | Windows 11 + WSL2 (Ubuntu 24.04)                           |
| CPU                | AMD Ryzen 9 7940HS                                         |
| GPU                | NVIDIA GeForce RTX 5060 Laptop GPU (8GB)                   |
| 内存               | 32GB DDR5                                                  |
| Python             | 3.11                                                       |
| PyTorch            | 2.12.0+cu130                                               |
| Ultralytics        | 8.4.60                                                     |
| CUDA               | 12.8 (WSL 专用版)                                          |

## 环境配置全过程

### 1. Windows 端：启用 WSL2 与安装 Ubuntu

以**管理员身份**打开 PowerShell，执行：

<pre><code class="lang-powershell">dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
</code></pre>

重启电脑后，安装 Ubuntu 24.04：

<pre><code class="lang-powershell">wsl --install -d Ubuntu-24.04
</code></pre>

首次启动 Ubuntu 时设置用户名和密码。

### 2. WSL 内安装 CUDA Toolkit 12.8（WSL 专用版）

进入 Ubuntu 终端，执行：

<pre><code class="lang-bash">wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
</code></pre>

验证：

<pre><code class="lang-bash">/usr/local/cuda-12.8/bin/nvcc --version
</code></pre>

### 3. 安装 Miniconda 与创建虚拟环境

<pre><code class="lang-bash">wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示操作，初始化 conda
source ~/.bashrc
conda create -n yolo_train python=3.11 -y
conda activate yolo_train
</code></pre>

### 4. 安装 PyTorch 与 Ultralytics

<pre><code class="lang-bash">pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python matplotlib
</code></pre>

> 若下载慢，可配置清华源：<code>pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple</code>

### 5. 环境验证与常见错误解决

验证 GPU：

<pre><code class="lang-bash">python -c "import torch; print(torch.cuda.is_available())"
</code></pre>

| 错误现象 | 原因 | 解决方法 |
|---------|------|----------|
| `conda: command not found` | conda 未初始化 | `source ~/.bashrc` |
| pip 超时 | 网络问题 | 配置国内镜像源 |
| `yolo train` 卡在 AMP 检查 | AMP 自动检测失败 | 训练命令加 `amp=False` |
| WSL 网络不通 | 虚拟交换机异常 | 重启电脑或 `wsl --shutdown` |
| `net stop LxssManager` 服务名无效 | 服务未正确安装 | 重启电脑 |

## 数据集准备

数据集来自 RoboFlow 上的 `Venom.v8i.yolov11`，包含 **12 个车辆类别**，已按 YOLO 格式划分 `train/`、`valid/`、`test/`。

- **验证集**：963 张图片，共 13450 个实例。
- **类别列表**（索引 0~11）：
  <pre><code>
  0: bus-l-         1: bus-s-         2: car
  3: extra large truck  4: large bus  5: medium truck
  6: small bus      7: small truck    8: truck-l-
  9: truck-m-      10: truck-s-      11: truck-xl-
  </code></pre>

**配置文件 `data.yaml`**：

<pre><code class="lang-yaml">path: ./
train: train/images
val: valid/images
test: test/images
nc: 12
names: ['bus-l-', 'bus-s-', 'car', 'extra large truck', 'large bus', 'medium truck', 'small bus', 'small truck', 'truck-l-', 'truck-m-', 'truck-s-', 'truck-xl-']
</code></pre>

## 模型训练

### 训练命令

<pre><code class="lang-bash">cd /path/to/dataset
conda activate yolo_train
yolo train data=data.yaml model=yolo11n.pt epochs=80 batch=16 imgsz=640 device=0 amp=False
</code></pre>

### 训练结果
- 最佳模型：`runs/detect/train/weights/best.pt` (5.5 MB)
- 训练曲线和混淆矩阵保存在 `runs/detect/train/`

### 训练中遇到的问题
- `WARNING: Download failure, retrying Arial.ttf` → 忽略，不影响训练。
- `CUDA out of memory` → 减小 `batch` 或 `imgsz`。
- Loss 变为 NaN → 关闭 AMP (`amp=False`)。

## 模型评估

验证集性能（部分类别）：

| Class               | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------------------- | ------ | --------- | --------- | ------ | ----- | -------- |
| car                 | 927    | 8537      | 0.853     | 0.745  | 0.817 | 0.497    |
| extra large truck   | 404    | 1162      | 0.765     | 0.558  | 0.638 | 0.412    |
| large bus           | 210    | 273       | 0.774     | 0.451  | 0.720 | 0.525    |
| small truck         | 517    | 1721      | 0.673     | 0.568  | 0.586 | 0.375    |
| **所有 12 类平均**  | 963    | 13450     | 0.494     | 0.611  | 0.441 | 0.308    |

**分析**：样本量多的类别（car）检测效果好；小样本类别（small bus）性能差，说明数据量对模型性能至关重要。

## 视频推理部署

### 基本命令

<pre><code class="lang-bash">yolo predict model=runs/detect/train/weights/best.pt source="video.mp4" save=True
</code></pre>

### 性能数据（RTX 5060）
- 预处理：1.3ms，推理：7.8ms，后处理：1.4ms
- 总耗时 ~10.5ms/帧 → **约 95 FPS**

### 高级选项
- 调整置信度：`conf=0.3`
- 只检测 car（类别索引 2）：`classes=2`
- 指定输出目录：`project=/path/to/output name=result`
- 流式推理：`stream=True`

## 项目复现指南

1. **克隆仓库**：
   <pre><code class="lang-bash">git clone git@github.com:Lzbfornepenthe/YOLOv11-Vehicle-Detection.git
   cd YOLOv11-Vehicle-Detection</code></pre>
2. **安装环境**：
   <pre><code class="lang-bash">conda env create -f environment.yml
   conda activate yolo_train</code></pre>
3. **准备数据集**：将数据集按 `train/images`, `train/labels`, `valid/images`, `valid/labels` 结构放置，修改 `data.yaml` 中的 `path`。
4. **训练**：
   <pre><code class="lang-bash">yolo train data=data.yaml model=yolo11n.pt epochs=80 batch=16 imgsz=640 device=0</code></pre>
5. **推理**：
   <pre><code class="lang-bash">yolo predict model=runs/detect/train/weights/best.pt source=demo.mp4 save=True</code></pre>

## 项目总结与改进方向

**成功点**：
- 完整搭建 WSL2 + CUDA 12.8 + PyTorch 环境，解决多个网络和路径问题。
- 训练 YOLOv11n 模型，汽车类 mAP50 达 0.817。
- 视频推理速度 95 FPS，满足实时要求。

**改进方向**：
- 增加数据扩充（夜间、雨天等场景）
- 使用更大模型（YOLOv11m/l）
- 导出 TensorRT 加速推理
- 提供一键训练/推理脚本

## 致谢与许可

- 感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供 YOLOv11 框架。
- 感谢 RoboFlow 社区提供数据集。

本项目采用 **MIT 许可证**。
