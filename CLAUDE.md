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
├── ball_image/                  # 小球原图、YOLO 数据集和合成中间产物
│   ├── source/                  # 原始轨道图
│   ├── datasets/                # yolo_dataset_10000/ref18/benchmark/smoke
│   └── artifacts/               # 按需生成的 output/debug 图，默认不保留
├── docs/                        # 项目布局与 K230 工作流文档
├── K230_Yolov5n/                # K230 边缘端部署
│   ├── main.py                  # K230 端推理入口 (检测+跟踪+串口)
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
├── server/                        # 远程服务器训练
│   ├── config.yaml                # 服务器连接配置 (SSH/路径/排除规则)
│   ├── setup.sh                   # 服务器端一键环境安装
│   ├── train.sh                   # 服务器端训练 (硬件自适应)
│   ├── sync_push.sh               # 本地→服务器 代码同步 (rsync)
│   ├── sync_pull.sh               # 服务器→本地 权重/结果拉取
│   └── connect.sh                 # 快速 SSH 连接
├── scripts/                       # 项目自动化脚本，不是 skills
│   ├── yolo/                      # 环境、训练、验证、推理、导出
│   │   ├── setup.ps1 / train.ps1 / validate.ps1
│   │   └── detect.ps1 / export.ps1
│   ├── dataset/                   # 数据生成与重组
│   └── k230/                      # K230 自动转换与配置
│       ├── convert.ps1
│       └── finish_training_and_convert.ps1
├── X-AnyLabeling/                 # 数据标注工具 (X-AnyLabeling)
│   ├── anylabeling/               # 核心应用代码
│   ├── assets/                    # 静态资源 (图标/图片)
│   ├── docs/                      # 使用文档
│   ├── examples/                  # 标注配置示例
│   └── requirements.txt           # 依赖清单
├── README.md                    # 工程使用说明
├── CLAUDE.md                    # 本文件 (Claude Code 自动加载)
└── .claude/skills/              # Claude Code 技能源副本
    ├── yolo-train/SKILL.md        # 本地训练
    ├── yolo-train-server/SKILL.md # 远程服务器训练
    ├── yolo-detect/SKILL.md       # 推理检测
    ├── yolo-export/SKILL.md       # 模型导出
    ├── yolo-validate/SKILL.md     # 验证评估
    └── yolo-kmodel/SKILL.md       # ONNX→K230 kmodel 转换
```

`.codex/skills/` 是上面 Claude skill 的 Codex 适配副本，具体用途见 `docs/PROJECT_LAYOUT.md`。

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

### 本地训练

```powershell
# 0. 数据标注 (可选)
cd X-AnyLabeling
pip install -r requirements.txt
python anylabeling/app.py

# 1. 安装环境
.\scripts\yolo\setup.ps1

# 2. 准备数据集 (当前基线集)
.\scripts\yolo\train.ps1 -Data ball_image\datasets\yolo_dataset_10000\data.yaml -Epochs 200

# 3. 训练
.\scripts\yolo\train.ps1 -Data your_data/datasets.yaml -Epochs 200

# 4. 检测
.\scripts\yolo\detect.ps1 -Source image.jpg -Weights yolov5-7.0\runs\train\ball_320\weights\best.pt

# 5. 导出 ONNX + 转 kmodel
.\scripts\yolo\export.ps1 -Weights runs/train/ball_320/weights/best.pt -Format onnx -ImgSize 320
.\scripts\k230\convert.ps1 -Model .\yolov5-7.0\runs\train\ball_320\weights\best.onnx -Dataset .\ball_image\datasets\yolo_dataset_10000\images\val
```

### 远程服务器训练

```bash
# 1. 编辑服务器连接配置
vim server/config.yaml

# 2. 推送代码 + 远程安装环境 (首次)
bash server/sync_push.sh
bash server/connect.sh    # SSH连接后执行: bash server/setup.sh

# 3. 训练 (代码同步 → 远程训练 → 拉取结果)
bash server/sync_push.sh
ssh -p <端口> root@<IP> "cd /root/YOLO && bash server/train.sh --data data/datasets.yaml --epochs 200"
bash server/sync_pull.sh
```

## 注意事项

1. 主工作目录是 `yolov5-7.0/`，数据根目录是 `ball_image/`。
2. K230 部署完整流程: `.pt → ONNX → kmodel → K230`；ONNX 只放在对应实验的 `runs/train/<experiment>/weights/`。
3. `yolo_dataset_10000` 是基线，`yolo_dataset_10000_ref18` 是对照实验，`benchmark/smoke` 只用于快速检查。
4. `.claude/skills/` 是 Claude 源副本，`.codex/skills/` 是 Codex 适配副本；完整用途见 `docs/PROJECT_LAYOUT.md`。
5. 所有脚本均在工程根目录执行。
6. Pillow 版本必须 < 10.0。
7. 字体文件 Arial.ttf 用于训练图表渲染。
