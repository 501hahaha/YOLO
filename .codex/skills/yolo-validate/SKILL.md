---
name: yolo-validate
description: YOLOv5 v7.0 模型验证与评估 — mAP/Precision/Recall/混淆矩阵。When the user wants to evaluate a trained model's performance on a validation dataset.
---

# YOLOv5 v7.0 模型验证与评估

## 项目环境

> 完整环境上下文见 ../../references/yolo-env.md（路径、Python、模型架构、输出约定）

- **框架**: YOLOv5 v7.0 (Ultralytics, GPL-3.0)
- **代码目录**: `yolov5-7.0/`
- **验证脚本**: `val.py`

## 命令模板

### 单模型验证（直接执行）

```bash
cd yolov5-7.0
python val.py \
  --weights <模型路径.pt> \
  --data <数据集yaml路径> \
  --imgsz 320 \
  --batch-size 16 \
  --verbose
```

### 多模型并行验证（推荐）

同时验证多个实验的 best.pt，使用 Codex 子任务并行执行，提示模板见 `../../references/yolo-validator.md`：

> 触发词: "validate all models in runs/train/" / "对比所有实验的 mAP"

1. 自动扫描 `runs/train/*/weights/best.pt`
2. 为每个模型创建一个独立的 Codex 子任务
3. N 个验证并行跑，节省 N× 等待时间
4. 结果按 mAP@0.5 排名汇总表格

**并行度**: 自动检测 GPU 数量。若 GPU 数 < 模型数，多余的 validator 切 CPU 避免争抢。

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

## 多模型对比（手动降级方案）

当不需要并行或只有 1-2 个模型时，可手动分别运行：

```bash
# 模型A
python val.py --weights runs/train/exp1/weights/best.pt --data data.yaml --name model_a

# 模型B
python val.py --weights runs/train/exp2/weights/best.pt --data data.yaml --name model_b
```

对比 `runs/val/model_a/` 和 `runs/val/model_b/` 中的结果。

## 在 Pipeline 中的位置

`yolo-pipeline` 技能在 Stage 2 自动调用并行验证，无需手动操作。参见 `../yolo-pipeline/SKILL.md`。

## 常见问题

1. **mAP很低**: 检查数据标注质量、类别数量、训练是否收敛
2. **某个类mAP为0**: 该类样本太少或标注有问题
3. **验证集图片加载失败**: 检查yaml中的路径是否正确
4. **batch size过大**: 已默认使用GPU内存允许的最大batch


