---
name: yolo-export
description: YOLOv5 v7.0 模型导出 — PyTorch→ONNX/TensorRT/TorchScript/TFLite。When the user wants to export a trained model for deployment (K230, edge devices, mobile).
---

# YOLOv5 v7.0 模型导出

## 项目环境

> 完整环境上下文见 ../../references/yolo-env.md（路径、Python、模型架构、输出约定）

- **框架**: YOLOv5 v7.0 (Ultralytics, GPL-3.0)
- **代码目录**: `yolov5-7.0/`
- **导出脚本**: `export.py`

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

### 多格式并行导出（推荐）

同时导出到多种格式，使用 Codex 子任务并行执行，提示模板见 `../../references/yolo-exporter.md`：

> 触发词: "export best.pt to ONNX, TensorRT, and TorchScript" / "同时导出多种格式"

1. 为每个目标格式创建一个独立的 Codex 子任务
2. N 个导出并行跑，省去串行等待
3. 结果统一汇总报告

**约束**: TensorRT 导出需 GPU，不可与 GPU 训练并行。

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

## K230 部署

> **自动化流程**: 使用 `yolo-pipeline` 技能可一次性完成 train → validate → export ONNX → convert kmodel 全流程。参见 `../yolo-pipeline/SKILL.md`。

手动步骤参考 `yolo-kmodel` 技能和 `K230_Yolov5n/` 目录。

## 常见问题

1. **ONNX导出失败**: 检查 `onnx` 和 `onnx-simplifier` 是否安装
2. **TensorRT导出失败**: 确认CUDA和TensorRT版本匹配
3. **K230转换失败**: 确保 `--batch 1`，`--imgsz` 与训练一致
4. **导出后精度下降**: 使用 `--simplify` 并检查opset版本


