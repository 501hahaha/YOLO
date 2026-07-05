---
name: yolo-train
description: YOLOv5 v7.0 模型训练 — 自定义数据集训练、继续训练、参数调优。When the user wants to train a YOLOv5 model, fine-tune, or adjust training hyperparameters.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5 v7.0 模型训练

## 项目环境

- **框架**: YOLOv5 v7.0 (Ultralytics, GPL-3.0)
- **代码目录**: `yolov5-7.0/`
- **Python环境**: 通过 `..\setup.ps1` 创建 `venv/`，或使用系统Python (推荐 venv)
- **关键依赖**: Pillow==9.5, torch>=1.10.0

## 模型架构

| 模型 | 大小 | 参数量 | 适用场景 |
|------|------|--------|----------|
| yolov5n | Nano | 1.9M | 移动端/K230部署 |
| yolov5s | Small | 7.2M | 通用检测 |
| yolov5m | Medium | 21.2M | 精度优先 |
| yolov5l | Large | 46.5M | 高精度 |
| yolov5x | XLarge | 86.7M | 最高精度 |

配置文件: `yolov5-7.0/models/yolov5n.yaml` (nc需改为实际类别数)

## 训练命令模板

### 1. 从头训练 (推荐)

```bash
cd yolov5-7.0
python train.py \
  --weights yolov5n.pt \
  --cfg models/yolov5n.yaml \
  --data <数据集yaml路径> \
  --epochs 200 \
  --batch-size 64 \
  --imgsz 320 \
  --device 0 \
  --workers 4 \
  --cache ram \
  --project runs/train \
  --name exp
```

### 2. 基于已有权重继续训练

```bash
python train.py \
  --weights best.pt \
  --data <数据集yaml路径> \
  --epochs 200 \
  --batch-size 16 \
  --imgsz 640 \
  --cache ram
```

### 3. 多GPU分布式训练

```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 \
  train.py --data data.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3
```

## 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--weights` | 预训练权重 | `yolov5n.pt` |
| `--cfg` | 模型配置 (仅从头训练需要) | `models/yolov5n.yaml` |
| `--data` | 数据集yaml | 必须指定 |
| `--epochs` | 训练轮数 | 100~500 |
| `--batch-size` | 批次大小 | 根据显存调整 |
| `--imgsz` | 输入尺寸 | 320/640 |
| `--device` | 设备 | `0` 或 `cpu` |
| `--cache` | 缓存图片到RAM | `ram` 加速训练 |
| `--resume` | 从断点恢复 | `True` |
| `--freeze` | 冻结骨干网络层数 | `10` |
| `--optimizer` | 优化器 | `SGD` / `Adam` |
| `--lr0` | 初始学习率 | `0.01` |

## 训练输出

```
runs/train/<exp名称>/
├── weights/
│   ├── last.pt       # 最后一轮
│   └── best.pt       # 最佳权重
├── results.csv       # 指标记录
├── results.png       # 指标曲线
├── train_batch*.jpg  # 训练批次样例
├── val_batch*.jpg    # 验证批次样例
└── labels.jpg        # 标签分布
```

## 超参数调优

超参数文件: `data/hyps/hyp.scratch-low.yaml`

建议：
- 小数据集/过拟合: 增大数据增强参数 (mosaic, hsv, degrees)
- 收敛慢: 增大 lr0，减小 warmup_epochs
- 类别不平衡: 调整 cls_pw (正样本权重)

## 常见问题

1. **CUDA out of memory**: 减小 `--batch-size` 或 `--imgsz`
2. **Pillow报错**: `pip install Pillow==9.5`
3. **字体报错**: 将字体文件放入 `./utils/fonts/`
4. **收敛慢**: 检查数据标注质量，增大epochs，调整lr0

## 数据集准备

参考 `data_template/` 目录:
1. 将图片放入 `images/`，标注放入 `labels/`
2. 运行 `python split.py` 分割数据集
3. 修改 `datasets/datasets.yaml` 中的 `nc` 和 `names`
4. 开始训练: `..\train.ps1 -Data data_template/datasets/datasets.yaml`

使用前检查对应的 `datasets/datasets.yaml` 确认类别数和路径。
