---
name: yolo-exporter
description: Exports a YOLOv5 PyTorch model to a single target format (ONNX, TensorRT, TorchScript, TFLite, etc.). Designed to be spawned as parallel sub-agents for multi-format simultaneous export.
allowed-tools: Read, Bash
---

# YOLOv5 Model Exporter (Sub-Agent)

导出**一个**模型到**一个**格式。N 个格式就 spawn N 个此 agent。

## 输入约定

父 skill 在 spawn 时传入：
- `MODEL_PATH` — .pt 权重路径
- `FORMAT` — 目标格式: `onnx` / `engine` / `torchscript` / `openvino` / `tflite` / `coreml` / `saved_model` / `tfjs`
- `IMG_SIZE` — 输入尺寸（默认 320）
- `BATCH` — batch size（K230 固定 1，其他选 1 或动态）
- `SIMPLIFY` — 是否简化 ONNX（默认 true）
- `DEVICE` — 设备（默认 `0`，TensorRT 必须 GPU）

## 执行

```bash
cd yolov5-7.0
python export.py \
  --weights <MODEL_PATH> \
  --include <FORMAT> \
  --imgsz <IMG_SIZE> \
  --batch <BATCH> \
  --device <DEVICE> \
  $( [ "<FORMAT>" = "onnx" ] && echo "--simplify --opset 12" )
```

## 输出

1. 导出文件路径（如 `best.onnx`, `best.engine`）
2. 文件大小
3. 警告/错误信息（如有）
