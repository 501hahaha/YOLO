---
name: yolo-detect
description: YOLOv5 v7.0 目标检测推理 — 单张/批量图片、视频、摄像头检测。When the user wants to run object detection inference with a trained YOLOv5 model.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5 v7.0 目标检测推理

## 项目环境

- **框架**: YOLOv5 v7.0 (Ultralytics, GPL-3.0)
- **代码目录**: `yolov5-7.0/`
- **Python环境**: 通过 `..\setup.ps1` 创建 `venv/`，或使用系统Python (推荐 venv)
- **推理脚本**: `yolov5-7.0/detect.py`

## 命令模板

### 单张图片检测

```bash
cd yolov5-7.0
python detect.py \
  --weights <权重路径.pt> \
  --source <图片路径> \
  --imgsz 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf \
  --project runs/detect \
  --name exp
```

### 批量图片/目录检测

```bash
python detect.py \
  --weights <权重路径.pt> \
  --source <目录路径> \
  --imgsz 640 \
  --device 0 \
  --save-txt \
  --save-conf
```

### 摄像头实时检测

```bash
python detect.py --weights <权重.pt> --source 0
```

### 视频文件检测

```bash
python detect.py --weights <权重.pt> --source video.mp4
```

### 支持多种模型格式

```bash
# PyTorch
python detect.py --weights best.pt --source img.jpg

# ONNX
python detect.py --weights best.onnx --source img.jpg

# TensorRT
python detect.py --weights best.engine --source img.jpg --device 0

# TorchScript
python detect.py --weights best.torchscript --source img.jpg
```

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weights` | 模型权重路径 | 必填 |
| `--source` | 输入源 (图片/目录/视频/摄像头/RTSP) | 必填 |
| `--imgsz` | 推理尺寸 | `640` |
| `--conf-thres` | 置信度阈值 | `0.25` |
| `--iou-thres` | NMS IOU阈值 | `0.45` |
| `--max-det` | 每张图最大检测数 | `1000` |
| `--device` | 设备 (CPU/GPU) | 自动 |
| `--save-txt` | 保存检测结果为txt | `False` |
| `--save-conf` | txt中附带置信度 | `False` |
| `--save-crop` | 保存裁剪的目标 | `False` |
| `--nosave` | 不保存结果图片 | `False` |
| `--classes` | 仅检测指定类别 (如 `0 2`) | 全部 |
| `--line-thickness` | 边界框线宽 | `3` |
| `--hide-labels` | 隐藏标签 | `False` |
| `--hide-conf` | 隐藏置信度 | `False` |

## 输出格式

检测结果保存至 `runs/detect/<exp名称>/`:
- 标注后的图片/视频
- `labels/` 目录 (若启用 `--save-txt`): 每张图对应一个 `.txt`，格式为 `class_id x_center y_center width height confidence`

## 常见问题

1. **置信度太低**: 适当提高 `--conf-thres`（如 `0.5`）
2. **漏检**: 降低 `--conf-thres`（如 `0.1`），检查图片尺寸是否匹配训练尺寸
3. **显存不足**: 减小 `--imgsz` 或切换到CPU
4. **ONNX推理失败**: 检查 `onnx` 和 `onnxruntime` 是否安装

## 项目可用权重

- `yolov5-7.0/best.pt` / `yolov5-7.0/best.onnx` - 训练后的最佳模型
- `yolov5-7.0/best_1000.pt` / `yolov5-7.0/best_1000.onnx` - 1000轮训练模型
- `yolov5n.pt` - 预训练基础权重
