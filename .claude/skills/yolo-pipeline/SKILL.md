---
name: yolo-pipeline
description: YOLOv5 end-to-end pipeline — train → validate → export ONNX → convert to K230 kmodel → verify. Automates the full workflow with sub-agent orchestration and parallel validation/export where independent.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

# YOLOv5 End-to-End Pipeline

将 train → validate → export → kmodel 全流程串联自动化。各阶段有顺序依赖（不可跳过上游），但阶段**内部**尽可能并行。

## 触发词

- "跑完整流程" / "train and deploy to K230"
- "从训练到部署一把梭" / "full pipeline"
- "一键训练导出部署"

## Pipeline 概览

```
Stage 0: 前置检查 (环境、数据集、校准集)
    │
Stage 1: 训练 (顺序, 单 GPU)
    │
Stage 2: 验证 (并行 — 多实验时 spawn N 个 yolo-validator)
    │
Stage 3: 导出 ONNX (可选多格式并行 export)
    │
Stage 4: 转 kmodel + 并行双重验证 (ONNX + kmodel 推理对比)
    │
Stage 5: 汇总报告
```

## 各阶段详情

### Stage 0: 前置检查 (Pre-flight)

1. 验证 Python 环境（venv 存在，torch 可 import）
2. 验证 dataset YAML 有效（路径存在，nc 与 names 一致）
3. 若有 kmodel 需求，检查校准数据集存在（≥ 20 张图片）
4. 检查必要依赖（onnx, onnxruntime 等）

### Stage 1: 训练

直接执行 `yolov5-7.0/train.py`：

```bash
cd yolov5-7.0
python train.py \
  --weights yolov5n.pt \
  --cfg models/yolov5n.yaml \
  --data <DATASET_YAML> \
  --epochs <EPOCHS> \
  --batch-size <BATCH_SIZE> \
  --imgsz <IMG_SIZE> \
  --device 0 \
  --workers 4 \
  --cache ram \
  --project runs/train \
  --name <EXP_NAME>
```

- 等待训练完成，捕获 `best.pt` 路径
- **失败处理**: 报告错误，询问是否从 `last.pt` 恢复

### Stage 2: 验证

- 若只训练了一个模型 → 直接验证 `best.pt`
- 若存在多个实验（`runs/train/*/weights/best.pt`） → **并行验证**:
  1. 扫描所有 `best.pt`
  2. 每个 spawn 一个 `agents/yolo-validator.md` 子代理
  3. 收集结果，按 mAP@0.5 排名
  4. 输出对比表格

参考: `skills/yolo-validate/SKILL.md`

**跳过**: 用户说 `--skip-validate` 时跳过此阶段

### Stage 3: 导出 ONNX

将 Stage 2 选出的最佳模型导出 ONNX：

```bash
cd yolov5-7.0
python export.py \
  --weights <BEST_PT> \
  --imgsz 320 \
  --batch 1 \
  --include onnx \
  --opset 12 \
  --simplify
```

**多格式并行**（可选）: 若用户指定多格式，spawn N 个 `agents/yolo-exporter.md` 子代理并行导出。

**跳过**: `--skip-export`

### Stage 4: 转 kmodel + 验证

1. 运行 `tools/to_kmodel.py` 量化编译
2. **并行双重验证**:
   - spawn `test_det_onnx.py`
   - spawn `test_det_kmodel.py`
   - 两者独立运行，对比检测结果
3. 报告余弦相似度：
   - \> 0.98: ✅ 精度损失极小
   - 0.95~0.98: ⚠️ 轻微损失，通常可用
   - 0.90~0.95: ⚠️ 明显损失，建议换 PTQOption 重试
   - < 0.90: ❌ 严重损失

**跳过**: `--skip-kmodel`

参考: `skills/yolo-kmodel/SKILL.md`

### Stage 5: 汇总报告

输出一览表：

| Stage | Status | Artifact | Metrics |
|-------|--------|----------|---------|
| Train | ✅ | `best.pt` | mAP@0.5 = ... |
| Validate | ✅ | `runs/val/...` | mAP@0.5 = ..., Recall = ... |
| Export | ✅ | `best.onnx` | size = ... MB |
| Kmodel | ✅ | `best.kmodel` | cosine = 0.98 |

## 可跳过标志

| 标志 | 跳过 |
|------|------|
| `--skip-validate` | Stage 2 |
| `--skip-export` | Stage 3 |
| `--skip-kmodel` | Stage 4 |

## 与单独技能的对应

此 pipeline 串联了以下技能（每个仍可独立使用）：

| Stage | 独立 Skill |
|-------|-----------|
| 训练 | `yolo-train` |
| 验证 | `yolo-validate` |
| 导出 | `yolo-export` |
| kmodel | `yolo-kmodel` |

## 常见问题

1. **Pipeline 中断后如何恢复**: 每个 Stage 的输出文件都在 `runs/` 下，可手动从断点继续
2. **GPU 并行争抢**: Stage 2 验证时，若模型数 > GPU 数，多余的 validator 自动切 CPU
3. **校准数据集不在默认位置**: 在 Stage 0 会提示用户指定路径
4. **只想导出不想训练**: 直接用 `yolo-export` 技能，无需走 pipeline
