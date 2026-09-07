---
name: yolo-label
description: X-AnyLabeling 数据标注工具 — YOLO 目标检测/实例分割标注、AI 辅助自动标注、标注格式转换。When the user wants to label images, use AI-assisted labeling, export YOLO format datasets, or set up the labeling workflow.
---

# X-AnyLabeling 数据标注工具

## 项目环境

- **工具**: X-AnyLabeling (开源, LGPL v3)
- **代码目录**: `X-AnyLabeling/`
- **Python要求**: Python 3.10+
- **平台**: Windows / Linux / macOS
- **仓库**: https://github.com/CVHub520/X-AnyLabeling

## 简介

X-AnyLabeling 是一款开源免费的 AI 数据标注工具，专为目标检测、实例分割、关键点检测等视觉任务设计。

**核心能力：**
- 🎯 **目标检测标注**：矩形框 (HBB)、旋转框 (OBB)、多边形 (SEG)
- 🤖 **AI 辅助标注**：内置 30+ 预训练模型 (YOLO/SAM/RT-DETR)，一键自动标注
- 📦 **格式兼容**：YOLO / COCO / VOC / LabelMe 双向导入导出
- 🔄 **迭代工作流**：手动标注 → 预训练 → AI自动标注 → 人工修正 → 正式训练

## 快速开始

### 安装

```bash
cd X-AnyLabeling

# 方式一：CPU 版本 (推荐入门)
pip install -r requirements.txt

# 方式二：GPU 版本 (AI 辅助标注更流畅)
pip install -r requirements-gpu.txt

# 启动应用
python anylabeling/app.py
```

### 标注流程 (YOLO 目标检测)

```
创建项目 → 导入图片 → 框选标注 → 导出 YOLO 格式 → 组织数据集 → 训练
```

#### Step 1: 创建项目

1. 启动后点击 **File → New Project**
2. 设置项目名称、任务类型选择 **Object Detection**
3. 添加所有需要检测的类别名称

#### Step 2: 导入图片

**File → Import Data** → 选择图片或图片目录。

#### Step 3: 手动标注

| 操作 | 快捷键 |
|------|--------|
| 矩形框标注 | **R** |
| 下一张 / 上一张 | **D** / **A** |
| 保存标注 | **Ctrl+S** |
| 完成当前图 | **F** |
| 删除标注 | **Del** |

#### Step 4: AI 辅助标注 (半自动)

1. 点击左侧 **AI** 按钮 (或 Ctrl+A)
2. 选择预训练模型
3. 点击 **▶ Run** 处理当前图片
4. 或点击 **"一次运行所有图片"** 批量处理
5. 手动修正 AI 标注不准确的地方

**内置推荐模型：**
| 任务 | 推荐模型 |
|------|----------|
| 通用检测 | YOLOv8s / YOLO11n |
| 实例分割 | SAM-HQ / Segment Anything |
| 旋转框 | YOLOv8n_obb |

#### Step 5: 导出 YOLO 格式

1. 在导出目录下创建 `classes.txt`，每行一个类别名，顺序与标注时一致
2. **File → Export Dataset** → 选择 **YOLO** 格式
3. 指定 `classes.txt` 路径
4. 勾选 **保存图片**，点击 **Export**

## 迭代标注工作流 (推荐)

```
① 手动标注 50~100 张
     ↓
② YOLOv5 预训练 (用小数据集初步训练)
     ↓
③ 导出 best.pt, 配置自定义模型到 X-AnyLabeling
     ↓
④ AI 一键自动标注剩余所有图片
     ↓
⑤ 逐张人工修正
     ↓
⑥ 导出完整数据集, 正式训练
```

### 自定义模型配置

训练出 `best.pt` 后，可将其导入 X-AnyLabeling 进行自动标注：

```yaml
type: yolov5          # 模型类型
name: my_model        # 模型名称
display_name: "我的自定义模型"
model_path: "best.onnx"
nms_threshold: 0.45
confidence_threshold: 0.4
classes:
  - class_name_1
  - class_name_2
```

将配置文件放到 `~/.xanylabeling_data/models/` 后重启即可。

## 数据集组织

导出后按 YOLO 标准结构组织：

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

### YOLO 标签格式 (.txt)

每行一个目标：`class_id x_center y_center width height`（归一化坐标）

## 数据集分割

使用工程自带的 `split.py` 按 8:1:1 比例自动分层抽样分割：

```bash
cd yolov5-7.0/data_template
# 1. 将标注好的图片放入 datasets/images/, 标签放入 datasets/labels/
# 2. 运行分割脚本
python split.py
```

## 常见问题

1. **启动后中文乱码**: 确认系统编码为 UTF-8
2. **AI 模型下载慢**: 手动下载后放入 `~/.xanylabeling_data/models/`
3. **导出缺少标签文件**: 检查 `classes.txt` 路径和类别数量
4. **自动标注效果差**: 先用更多样本训练自定义模型，或调低 `confidence_threshold`
5. **大幅面图片卡顿**: 先缩放到 1920px 以内

## 参考资源

- 官方仓库: https://github.com/CVHub520/X-AnyLabeling
- 自定义模型: https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/custom_model.md


