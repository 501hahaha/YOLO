---
name: yolo-project
description: YOLOV 工程概览 — 项目结构、环境配置、数据集、部署目标 (K230)。Automatically loaded on startup to provide project context.
---

# YOLOV 工程概览

## 项目简介

基于 YOLOv5 v7.0 的目标检测工程，面向 **K230 边缘设备** 部署，30MB 轻量可移植。

## 目录结构

```
YOLO/
├── yolov5-7.0/                  # YOLOv5 v7.0 核心代码
│   ├── train.py                 # 训练脚本 (检测任务)
│   ├── detect.py                # 推理脚本
│   ├── val.py                   # 验证/评估脚本
│   ├── export.py                # 模型导出脚本
│   ├── classify/                # 分类任务 (train/predict/val)
│   ├── segment/                 # 分割任务
│   ├── models/                  # 模型配置 (yolov5n/s/m/l/x.yaml)
│   ├── data/                    # 数据集配置 + 超参数 (hyps/)
│   ├── data_template/           # 数据集目录模板 (含 split.py)
│   ├── utils/                   # 工具函数库
│   └── Arial.ttf / Arial.zip    # 字体文件 (图表渲染用)
├── K230_Yolov5n/                # K230 边缘端部署
│   ├── convert.ps1              # 一键 ONNX→kmodel 转换脚本
│   ├── main.py                  # K230 端推理入口 (检测+跟踪+串口)
│   ├── best.kmodel              # 已编译 K230 模型
│   ├── nncase_kpu-*.whl         # Windows nncase KPU 离线安装包
│   ├── tools/                   # kmodel 转换工具集
│   │   ├── to_kmodel.py         # ONNX → kmodel
│   │   ├── simulate.py          # kmodel vs ONNX 精度对比
│   │   ├── save_bin.py          # 生成测试输入
│   │   ├── test_det_onnx.py     # ONNX 推理测试
│   │   └── test_det_kmodel.py   # kmodel 推理测试
│   ├── mp_deployment_source/    # 部署源码和 kmodel
│   ├── CV_test/                 # 端侧测试脚本
│   └── test_yolov5/             # 分类/检测/分割测试
├── setup.ps1                    # 一键安装环境 (自动检测GPU/CPU)
├── train.ps1                    # 训练 (硬件自适应 batch-size/workers)
├── detect.ps1                   # 推理检测
├── export.ps1                   # 模型导出
├── validate.ps1                 # 验证评估
├── README.md                    # 工程使用说明
├── CLAUDE.md                    # 本文件 (Claude Code 自动加载)
├── 模型训练.md                   # 训练完整指南
└── .claude/skills/              # Claude Code 技能文件
    ├── yolo-train/SKILL.md      # 训练
    ├── yolo-detect/SKILL.md     # 推理检测
    ├── yolo-export/SKILL.md     # 模型导出
    └── yolo-validate/SKILL.md   # 验证评估
```

## 环境

| 项目 | 配置 |
|------|------|
| Python | `setup.ps1` 自动创建 `venv/` 并安装依赖 |
| 框架 | YOLOv5 v7.0 (PyTorch) |
| 关键依赖 | Pillow==9.5, torch>=1.10.0 |
| 部署目标 | K230 (Kendryte, nncase KPU) |
| 模型 | yolov5n (Nano, 1.9M参数) |
| 输入尺寸 | 320x320 (K230适配) |

## 快速开始

```powershell
# 1. 安装环境
.\setup.ps1

# 2. 准备数据集 (参考 data_template/ 目录结构)

# 3. 训练
.\train.ps1 -Data your_data/datasets.yaml -Epochs 200

# 4. 检测
.\detect.ps1 -Source image.jpg -Weights runs/train/exp/weights/best.pt

# 5. 导出 ONNX + 转 kmodel
.\export.ps1 -Weights best.pt -Format onnx
cd K230_Yolov5n
.\convert.ps1 -Model ..\yolov5-7.0\best.onnx -Dataset <校准集目录>
```

## 注意事项

1. 主工作目录是 `yolov5-7.0/`
2. K230 部署完整流程: `.pt → ONNX → kmodel → K230`
3. 所有脚本均在工程根目录执行
4. Pillow 版本必须 < 10.0
5. 字体文件 Arial.ttf 用于训练图表渲染
