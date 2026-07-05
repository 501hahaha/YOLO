---
name: yolo-validate
description: YOLOv5 v7.0 模型验证与评估 — mAP/Precision/Recall/混淆矩阵。When the user wants to evaluate a trained model's performance on a validation dataset.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5 v7.0 模型验证与评估

## 项目环境

- **框架**: YOLOv5 v7.0
- **代码目录**: `yolov5-7.0/`
- **Python环境**: 通过 `..\setup.ps1` 创建 `venv/`，或使用系统Python (推荐 venv)
- **验证脚本**: `yolov5-7.0/val.py`

## 命令模板

### 基础验证

```bash
cd yolov5-7.0
python val.py \
  --weights <模型路径.pt> \
  --data <数据集yaml路径> \
  --imgsz 320 \
  --batch-size 16 \
  --verbose
```

### 使用ONNX/Engine验证

```bash
# ONNX
python val.py --weights best.onnx --data data.yaml --imgsz 320

# TensorRT
python val.py --weights best.engine --data data.yaml --imgsz 640 --device 0
```

### TTA (测试时增强) 验证

```bash
python val.py --weights best.pt --data data.yaml --imgsz 640 --augment
```

### 多尺寸验证

```bash
python val.py --weights best.pt --data data.yaml --imgsz 320 416 640
```

## 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--weights` | 模型权重 | 必填 |
| `--data` | 数据集yaml | 必填 |
| `--imgsz` | 验证尺寸 | 与训练一致 |
| `--batch-size` | 批次大小 | `16` |
| `--conf-thres` | 置信度阈值 | `0.001` |
| `--iou-thres` | NMS IOU阈值 | `0.6` |
| `--max-det` | 最大检测数 | `300` |
| `--task` | 任务类型 | `val` / `test` |
| `--device` | 设备 | `0` 或 `cpu` |
| `--verbose` | 输出每类指标 | 建议开启 |
| `--save-hybrid` | 保存混合标签 | 再次训练用 |
| `--save-json` | 保存COCO JSON | 提交比赛用 |
| `--augment` | TTA增强 | 提分 (~1-2 mAP) |
| `--plots` | 生成混淆矩阵等图表 | 建议开启 |

## 评估指标

验证输出包含以下指标:

- **mAP@0.5**: IoU=0.5时的平均精度 (主要参考)
- **mAP@0.5:0.95**: IoU从0.5到0.95的平均精度 (COCO标准)
- **Precision**: 精确率 = TP / (TP + FP)
- **Recall**: 召回率 = TP / (TP + FN)

## 输出文件

```
runs/val/<exp名称>/
├── confusion_matrix.png  # 混淆矩阵 (--plots)
├── precision_recall.png  # P-R曲线
├── F1_curve.png          # F1-置信度曲线
├── labels.jpg            # 标签分布
└── results.txt           # 详细验证结果
```

## 多模型对比

为了对比不同模型的性能，分别运行验证并记录:

```bash
# 模型A
python val.py --weights runs/train/exp1/weights/best.pt --data data.yaml --name model_a

# 模型B
python val.py --weights runs/train/exp2/weights/best.pt --data data.yaml --name model_b
```

对比 `runs/val/model_a/` 和 `runs/val/model_b/` 中的结果。

## 常见问题

1. **mAP很低**: 检查数据标注质量、类别数量、训练是否收敛
2. **某个类mAP为0**: 该类样本太少或标注有问题
3. **验证集图片加载失败**: 检查yaml中的路径是否正确
4. **batch size过大**: 已默认使用GPU内存允许的最大batch
