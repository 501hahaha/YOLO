# YOLOv5 目标检测工程

基于 [YOLOv5 v7.0](https://github.com/ultralytics/yolov5) 的目标检测训练/部署工程。

## 目录结构

```
YOLO/
├── yolov5-7.0/              # YOLOv5 核心代码
│   ├── train.py             # 训练脚本
│   ├── detect.py            # 推理脚本
│   ├── val.py               # 验证脚本
│   ├── export.py            # 导出脚本 (ONNX/TensorRT)
│   ├── models/              # 模型配置文件 (yolov5n/s/m/l/x)
│   ├── data/                # 示例数据 + 超参数模板
│   ├── data_template/       # 数据集目录结构模板 (含 split.py)
│   ├── utils/               # 工具库
│   ├── classify/            # 分类任务
│   └── segment/             # 分割任务
├── K230_Yolov5n/            # K230 边缘端部署
│   ├── main.py              # K230 推理入口
│   ├── mp_deployment_source/ # 部署源码和 kmodel
│   ├── CV_test/             # 端侧测试脚本
│   └── test_yolov5/         # detect/classify/segment 测试
├── ball_image/              # 原始图和 YOLO 数据集
│   ├── source/              # 原始轨道图
│   └── datasets/            # 正式、对照、benchmark、smoke 数据集
├── docs/                    # 项目文档
│   ├── YOLO_TRAINING.md     # 详细训练指南
│   └── YOLO_K230_WORKFLOW.md # K230 工作流
├── scripts/                 # 项目自动化脚本，不是 skills
│   ├── yolo/                # 环境、训练、验证、推理、导出
│   │   ├── setup.ps1
│   │   ├── train.ps1
│   │   ├── validate.ps1
│   │   ├── detect.ps1
│   │   └── export.ps1
│   ├── dataset/             # 数据生成脚本
│   └── k230/                # K230 转换与流水线脚本
├── README.md                # 本文件
└── CLAUDE.md                # Claude Code 工程概览
```

## 快速开始

### 1. 安装环境

```powershell
.\scripts\yolo\setup.ps1
```

自动完成：
- 创建 Python 虚拟环境 (`venv/`)
- 检测 GPU/CPU，安装对应 PyTorch 版本
- 安装 YOLOv5 所有依赖

### 2. 准备数据集

按 YOLO 格式组织数据集：

```
your_dataset/
├── datasets.yaml    # 配置文件 (见下方模板)
├── train/
│   ├── images/      # 训练图片
│   └── labels/      # 训练标签 (.txt)
└── val/
    ├── images/      # 验证图片
    └── labels/      # 验证标签
```

**datasets.yaml 模板**：

```yaml
path: ./your_dataset/    # 数据集根目录
train: train.txt          # 或直接用 images/ 路径
val: val.txt
nc: 1                     # 类别数
names:
  0: class_name
```

### 3. 训练

```powershell
.\scripts\yolo\train.ps1 -Data your_dataset/datasets.yaml
```

脚本会自动检测硬件并选择最优 batch size / workers：

| 显存 | 自动 batch size |
|------|----------------|
| ≥24GB | 128 |
| ≥12GB | 64 |
| ≥8GB | 32 |
| ≥4GB | 16 |
| <4GB / CPU | 8 |

其他参数：

```powershell
# 自定义轮数和尺寸
.\scripts\yolo\train.ps1 -Data data.yaml -Epochs 300 -ImgSize 640

# 继续训练
.\scripts\yolo\train.ps1 -Data data.yaml -Weights best.pt -Epochs 100

# 从断点恢复
.\scripts\yolo\train.ps1 -Data data.yaml -Resume
```

### 4. 检测

```powershell
# 单张图
.\scripts\yolo\detect.ps1 -Source image.jpg -Weights runs/train/exp/weights/best.pt

# 整个目录 (保存 txt 结果)
.\scripts\yolo\detect.ps1 -Source images/ -Weights best.pt -SaveTxt

# 调低置信度阈值
.\scripts\yolo\detect.ps1 -Source image.jpg -Conf 0.1
```

### 5. 导出

```powershell
.\scripts\yolo\export.ps1 -Weights best.pt -Format onnx -ImgSize 320
```

### 6. 验证

```powershell
.\scripts\yolo\validate.ps1 -Weights best.pt -Data data.yaml
```

## 模型选择

| 模型 | 参数量 | FLOPs | 速度 | 适用 |
|------|--------|-------|------|------|
| yolov5n | 1.9M | 4.2G | 快 | 移动端/边缘 |
| yolov5s | 7.2M | 16.5G | 较快 | 通用 |
| yolov5m | 21.2M | 49G | 中等 | 精度优先 |
| yolov5l | 46.5M | 109G | 较慢 | 高精度 |
| yolov5x | 86.7M | 206G | 慢 | 最高精度 |

## 跨主机使用

整个工程不依赖硬编码路径，复制到任意 Windows 主机后：

```powershell
.\scripts\yolo\setup.ps1    # 自动检测硬件 + 安装环境
.\scripts\yolo\train.ps1 -Data your_data.yaml
```

- **GPU 主机**: 自动安装 CUDA 版 PyTorch
- **CPU 主机**: 自动安装 CPU 版 PyTorch
- **不同显存**: 自动调整 batch size

## 常用问题

| 问题 | 解决 |
|------|------|
| CUDA out of memory | 减小 `-BatchSize` 或 `-ImgSize` |
| 训练太慢 | 加 `-NoCache` 关闭 RAM 缓存节省内存 |
| 收敛不佳 | 增加 `-Epochs`，检查标注质量 |
| 找不到数据集 | 确保 yaml 中 path 使用相对路径 |

## 开源协议

本项目原创代码采用 GNU General Public License v3.0（GPL-3.0）许可证，完整文本见根目录 [`LICENSE`](LICENSE)。

仓库内的 `yolov5-7.0/` 和 `X-AnyLabeling/` 是第三方组件，继续遵循各自目录中的许可证和版权声明。
