# YOLOv5 模型训练完整指南

> 📅 更新日期：2026-02-26
> 🎯 用途：完整的YOLOv5训练、验证、推理和部署流程

---

## 1. 环境准备与注意事项

在开始训练前，请确保满足以下环境和依赖要求，以避免常见报错：

### 1.1 关键依赖版本

- **Pillow版本限制**：必须小于 `10.0`（推荐安装 `9.5`），否则训练结束后的输出效果图片可能会失败。
  ```bash
  pip install Pillow==9.5
  ```

- **PyTorch版本**：建议使用 `1.10.0` 或更高版本
  ```bash
  pip install torch>=1.10.0
  ```

- **其他依赖**：根据 `requirements.txt` 安装
  ```bash
  pip install -r requirements.txt
  ```

### 1.2 常见问题处理

- **权重加载参数**：加载权重时建议附带参数 `weights_only=False`。

- **字体问题处理**：若出现字体转换报错，需下载并安装字体包，并将其存放至 `./utils/fonts` 目录下，以确保训练完成后数据结尾显示正常。

---

## 2. 数据准备阶段

### 2.1 数据集目录结构

推荐的标准YOLO数据集结构：

```
datasets/
├── images/
│   ├── train/      # 训练集图片
│   ├── val/        # 验证集图片
│   └── test/       # 测试集图片（可选）
├── labels/
│   ├── train/      # 训练集标签（.txt格式）
│   ├── val/        # 验证集标签
│   └── test/       # 测试集标签
└── datasets.yaml   # 配置文件
```

### 2.2 关键文件说明

- **`classes.txt`**：包含标签内容的文件，每行一个类别名。
- **`datasets.yaml`**：数据集配置文件，需指定路径、类别数和类别名。

### 2.3 数据分割

使用脚本 `split.py` 将整理好的文件按比例分割：

```bash
python split.py --source ./data --output ./datasets --ratio 0.8 0.1 0.1
```

**参数说明**：
- `--source`: 原始数据路径
- `--output`: 输出路径
- `--ratio`: 训练/验证/测试集比例

---

## 3. YOLOv5 训练流程

### 3.1 训练命令示例

#### 示例 1：基于已有权重继续训练

```bash
python train.py \
  --data data_test_V2/datasets/datasets.yaml \
  --weights best.pt \
  --img 640 \
  --epochs 200 \
  --batch-size 16 \
  --cache ram \
  --project runs/train \
  --name exp_continue
```

#### 示例 2：从头训练YOLOv5n

```bash
python train.py \
  --weights yolov5n.pt \
  --cfg models/yolov5n.yaml \
  --data datasets/fruits_yolo.yaml \
  --epochs 300 \
  --batch-size 8 \
  --imgsz 320 \
  --device '0' \
  --workers 4 \
  --optimizer SGD
```

### 3.2 训练参数说明

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `--weights` | 预训练权重文件 | `yolov5n.pt`, `best.pt` |
| `--cfg` | 模型配置文件 | `models/yolov5n.yaml` |
| `--data` | 数据集配置文件 | `datasets.yaml` |
| `--epochs` | 训练轮数 | `100`, `200`, `300` |
| `--batch-size` | 批次大小 | `8`, `16`, `32` |
| `--imgsz` | 图片尺寸 | `320`, `640`, `1280` |
| `--device` | 设备选择 | `'0'` (GPU 0), `cpu` |
| `--cache` | 数据缓存 | `ram`, `disk` |
| `--workers` | 数据加载线程数 | `4`, `8` |

### 3.3 训练监控

训练过程中会生成以下内容：

```
runs/train/exp/
├── weights/
│   ├── last.pt       # 最后一轮权重
│   └── best.pt       # 最佳权重（基于验证集指标）
├── results.csv       # 训练指标记录
├── train_batch0.jpg  # 训练批次可视化
└── labels.jpg        # 标签分布图
```

---

## 4. 模型验证与测试

### 4.1 验证命令

```bash
python val.py \
  --weights runs/train/exp/weights/best.pt \
  --data datasets/fruits_yolo.yaml \
  --imgsz 320 \
  --batch-size 16 \
  --verbose
```

### 4.2 预测/检测命令

#### 单张图片检测

```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source path/to/image.jpg \
  --imgsz 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf
```

#### 批量图片检测

```bash
python detect.py \
  --weights path/to/best.pt \
  --source path/to/test/images/ \
  --imgsz 640 \
  --device '0' \
  --save-txt \
  --nosave
```

### 4.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--conf-thres` | 置信度阈值 | `0.25` |
| `--iou-thres` | IOU阈值 | `0.45` |
| `--save-txt` | 保存检测结果为txt | `False` |
| `--save-conf` | 在txt中保存置信度 | `False` |
| `--nosave` | 不保存图片 | `False` |

---

## 5. 模型导出与部署

### 5.1 导出为ONNX格式

```bash
python export.py \
  --weights runs/train/exp/weights/best.pt \
  --imgsz 320 \
  --batch 1 \
  --include onnx \
  --opset 12
```

### 5.2 导出为TensorRT格式

```bash
python export.py \
  --weights best.pt \
  --imgsz 640 \
  --batch 1 \
  --include engine \
  --device '0'
```

### 5.3 支持的导出格式

```bash
# 导出多种格式
python export.py \
  --weights best.pt \
  --include torchscript onnx openvino engine
```

### 5.4 导出格式说明

| 格式 | 用途 | 适用场景 |
|------|------|----------|
| `torchscript` | PyTorch序列化 | Python部署 |
| `onnx` | 跨平台模型 | 多框架兼容 |
| `openvino` | Intel优化 | CPU推理加速 |
| `engine` | TensorRT | NVIDIA GPU加速 |
| `tflite` | TensorFlow Lite | 移动端部署 |

---

## 6. 常见问题排查

### 6.1 训练过程问题

**问题1：CUDA out of memory**
- 解决方案：减小 `batch-size` 或 `imgsz`
- 或使用 `--batch-size -1` 让程序自动计算最大batch

**问题2：收敛慢或不收敛**
- 检查数据标注质量
- 调整学习率 `--lr0 0.01`
- 增加训练轮数 `--epochs 500`

### 6.2 预测过程问题

**问题1：预测结果为空**
- 降低置信度阈值 `--conf-thres 0.1`
- 检查类别名是否匹配

**问题2：预测速度慢**
- 减小图片尺寸 `--imgsz 320`
- 使用导出格式（如TensorRT）

---

## 7. 快速参考

### 常用命令速查

```bash
# 1. 快速训练
python train.py --data data.yaml --weights yolov5s.pt --epochs 100

# 2. 验证模型
python val.py --weights best.pt --data data.yaml

# 3. 批量检测
python detect.py --weights best.pt --source images/ --save-txt

# 4. 导出ONNX
python export.py --weights best.pt --include onnx
```

---

## 8. 最佳实践

1. **数据质量优先**：确保标注准确、类别平衡
2. **从小模型开始**：先用 `yolov5n` 测试，再升级到 `yolov5s/m/l/x`
3. **保存中间结果**：定期备份 `best.pt` 和训练日志
4. **可视化验证**：使用 `detect.py` 查看实际检测效果
5. **性能对比**：记录不同参数配置下的mAP、FPS指标

python train.py --data data_test_V3/datasets/datasets.yaml --weights yolov5n.pt --img 320 --epochs 200 --batch-size 64 --cache ram  
# Kmodel转换
先导出onnx模型，再转换成kmodel
python export.py --weight runs/train/<exp名称>/weights/best.pt --imgsz 320 --batch 1 --include onnx

python to_kmodel.py --target k230 --model <onnx路径> --dataset <校准集目录> --input_width 320 --input_height 320 --ptq_option 0

python train.py  --data <数据集路径>/datasets.yaml --weights yolov5n.pt --img 320 --epochs 200 --batch-size 64 --cache ram