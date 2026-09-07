---
name: yolo-validator
description: Validates a single YOLOv5 model (.pt or .onnx) against a dataset. Designed as a Codex subtask prompt template for parallel multi-model comparison.
---

# YOLOv5 Model Validator (Codex Sub-Agent Template)

验证**一个**模型。并行化由父 skill 创建 N 个 Codex 子任务实现。

## 输入约定

父 skill 或主代理在创建子任务时传入：
- `MODEL_PATH` — .pt 或 .onnx 权重路径
- `DATA_YAML` — 数据集 yaml 路径
- `IMG_SIZE` — 输入尺寸（默认 320）
- `EXP_NAME` — 实验名（用于 runs/val/ 子目录）
- `BATCH_SIZE` — 批次大小（默认 16）
- `DEVICE` — 设备（默认自动检测。若并行数量 > GPU 数，强制使用 CPU 避免争抢）

## 执行

```bash
cd yolov5-7.0
python val.py \
  --weights <MODEL_PATH> \
  --data <DATA_YAML> \
  --imgsz <IMG_SIZE> \
  --batch-size <BATCH_SIZE> \
  --device <DEVICE> \
  --project runs/val \
  --name <EXP_NAME> \
  --exist-ok \
  --verbose
```

## 输出

汇总以下信息：
1. **mAP@0.5** 和 **mAP@0.5:0.95**
2. **Precision** 和 **Recall**
3. 各类别 AP（若 `--verbose`）
4. 结果路径: `runs/val/EXP_NAME/`
5. 模型路径: `MODEL_PATH`

若为 ONNX 模型，额外报告：
- ONNX vs PT 精度差异（若同时有 .pt 对照）

