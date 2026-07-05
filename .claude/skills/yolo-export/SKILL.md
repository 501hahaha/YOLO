---
name: yolo-export
description: YOLOv5 v7.0 模型导出 — PyTorch→ONNX/TensorRT/TorchScript/TFLite。When the user wants to export a trained model for deployment (K230, edge devices, mobile).
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5 v7.0 模型导出

## 项目环境

- **框架**: YOLOv5 v7.0
- **代码目录**: `yolov5-7.0/`
- **Python环境**: 通过 `..\setup.ps1` 创建 `venv/`，或使用系统Python (推荐 venv)
- **导出脚本**: `yolov5-7.0/export.py`

## 支持的导出格式

| 格式 | `--include` | 输出 | 适用场景 |
|------|-------------|------|----------|
| PyTorch | (默认) | `.pt` | Python部署 |
| TorchScript | `torchscript` | `.torchscript` | C++ LibTorch |
| ONNX | `onnx` | `.onnx` | 跨平台/转Kmodel |
| OpenVINO | `openvino` | `_openvino_model/` | Intel CPU加速 |
| TensorRT | `engine` | `.engine` | NVIDIA GPU加速 |
| CoreML | `coreml` | `.mlmodel` | Apple设备 |
| TFLite | `tflite` | `.tflite` | 移动端/嵌入式 |
| TF SavedModel | `saved_model` | `_saved_model/` | TensorFlow Serving |
| TF.js | `tfjs` | `_web_model/` | 浏览器端 |

## 命令模板

### 导出ONNX (最常用，K230部署必需)

```bash
cd yolov5-7.0
python export.py \
  --weights <模型路径.pt> \
  --imgsz 320 \
  --batch 1 \
  --include onnx \
  --opset 12 \
  --simplify
```

### 导出TensorRT

```bash
python export.py \
  --weights best.pt \
  --imgsz 640 \
  --batch 1 \
  --include engine \
  --device 0
```

### 批量导出多种格式

```bash
python export.py \
  --weights best.pt \
  --include torchscript onnx openvino engine
```

### 动态batch导出

```bash
python export.py --weights best.pt --include onnx --dynamic
```

## 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--weights` | 输入权重 | 必填 |
| `--imgsz` | 导出尺寸 | 与训练一致 |
| `--batch` | 导出batch size | `1` (K230要求) |
| `--include` | 目标格式 | `onnx`, `engine` 等 |
| `--opset` | ONNX算子集版本 | `12` |
| `--simplify` | ONNX模型简化 | 建议开启 |
| `--dynamic` | 动态batch/尺寸 | 服务器用 |
| `--half` | FP16半精度 | GPU部署 |
| `--int8` | INT8量化 | 边缘设备 |
| `--device` | 导出设备 | `0` 或 `cpu` |
| `--topk-per-class` | 每类最多检测数 | `100` |
| `--iou-thres` | NMS IOU阈值 | `0.45` |
| `--conf-thres` | NMS置信度阈值 | `0.25` |

## K230部署完整流程

1. **训练模型**: 使用 `yolo-train` skill
2. **导出ONNX**: 见上方命令，确保 `--imgsz 320 --batch 1`
3. **转Kmodel**: 使用nncase工具转换
   ```bash
   python to_kmodel.py \
     --target k230 \
     --model best.onnx \
     --dataset <校准数据集路径> \
     --input_width 320 \
     --input_height 320 \
     --ptq_option 0
   ```
4. **部署**: 将 `.kmodel` 文件部署到K230设备

参考: `K230_Yolov5n/` 目录下的部署文件

## 常见问题

1. **ONNX导出失败**: 检查 `onnx` 和 `onnx-simplifier` 是否安装
2. **TensorRT导出失败**: 确认CUDA和TensorRT版本匹配
3. **K230转换失败**: 确保 `--batch 1`，`--imgsz` 与训练一致
4. **导出后精度下降**: 使用 `--simplify` 并检查opset版本
