---
name: yolo-env
description: Shared environment context for this YOLOv5 and K230 project, including paths, the local Conda interpreter, dataset, and output contracts.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5/K230 项目环境

## 路径

| 路径 | 用途 |
|---|---|
| `yolov5-7.0/` | YOLOv5 v7.0 核心代码、训练、验证和导出 |
| `ball_image/datasets/yolo_dataset_10000/` | 10,000 张小球数据和 YOLO 标注 |
| `K230_Yolov5n/` | K230 kmodel 转换脚本、参考部署代码和离线 wheel |
| `docs/YOLO_K230_WORKFLOW.md` | 当前项目的训练/转换流程 |
| `.codex/skills/` | Codex 项目 skills |

## Python 环境

训练和转换使用：

```text
.conda/yolov5_train_py310/python.exe
```

已验证训练侧使用 Python 3.10、PyTorch CUDA 12.8、RTX 5070 Ti Laptop GPU；K230 转换需要在该环境额外安装 `nncase==2.9.0`、`nncase-kpu` wheel、`onnx==1.17.0`、`onnxruntime==1.19.0`、`onnxsim==0.4.36`、`onnxscript==0.6.2`、`onnx_ir==1.0.0`、`ml_dtypes==0.5.1`、`numpy==1.26.4` 和 `scipy==1.14.1`。系统 dotnet 为 7.0.410；`yolov5-7.0/export.py` 使用 legacy exporter 兼容 PyTorch 2.11。
Windows nncase 还需要项目环境 `Lib/site-packages/libomp140.x86_64.dll`。

## 当前模型契约

- 模型：YOLOv5n 检测，类别 `ball`。
- 输入：训练、导出和 K230 均为 `320×320`；导出 batch 为 `1`。
- ONNX：静态 shape，opset `12`，不使用 dynamic。
- K230：`target=k230`、`NCHW`、uint8 前处理和 20 张图片校准，默认 `PTQOption=0`。

## 输出

正式训练输出：

```text
yolov5-7.0/runs/train/ball_320/weights/best.pt
yolov5-7.0/runs/train/ball_320/results.csv
```

导出的 `best.onnx` 和 `best.kmodel` 与 checkpoint 位于同一 `weights/` 目录。不要覆盖 `K230_Yolov5n/mp_deployment_source/` 中已有的模型；新模型的 `deploy_config.json` 需要使用最终 checkpoint 实际 anchors，并将类别设置为 `ball`。
