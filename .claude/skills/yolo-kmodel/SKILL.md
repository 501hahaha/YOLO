---
name: yolo-kmodel
description: YOLOv5 ONNX → K230 kmodel 转换 — nncase 量化编译、精度验证、端侧推理测试。When the user wants to convert a YOLOv5 ONNX model to K230 kmodel format, verify precision, or test kmodel inference.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# YOLOv5 ONNX → K230 kmodel 转换

## 项目环境

- **部署目录**: `K230_Yolov5n/`
- **工具目录**: `K230_Yolov5n/tools/`
- **一键脚本**: `K230_Yolov5n/convert.ps1`
- **Python环境**: 需要安装 `nncase` (KPU 编译工具) 和 `onnx` / `onnxruntime` / `onnxsim`

## 转换流程概览

```
.pt 模型  ──[yolo-export]──▶  ONNX  ──[yolo-kmodel]──▶  kmodel  ──▶  K230 部署
                                                          │
                                                          ├── 精度验证 (simulate.py)
                                                          └── 推理测试 (test_det_kmodel.py)
```

## 前置条件

1. **ONNX 模型已导出**（通过 `yolo-export` skill），确保：
   - `--imgsz 320`（K230 适配尺寸）
   - `--batch 1`（K230 单 batch）
   - `--opset 12`
   - `--simplify` 已开启

2. **校准数据集**：从训练集随机抽取 20 张图片，单独放入一个目录

3. **nncase 环境**：

   **所有平台通用依赖**：
   ```bash
   pip install onnx
   pip install onnxruntime
   pip install onnxsim
   ```

   **Linux 平台**（nncase 和 nncase-kpu 均在线安装）：
   ```bash
   # 安装 dotnet-7（nncase 2.x 依赖）
   sudo apt-get install -y dotnet-sdk-7.0

   # 安装 nncase 及 KPU 插件
   pip install --upgrade pip
   pip install nncase==2.9.0
   pip install nncase-kpu==2.9.0
   ```

   **Windows 平台**（nncase 可在线安装，nncase-kpu 必须离线安装）：
   ```powershell
   # 1. 下载安装 dotnet-7 runtime 并添加环境变量
   #    https://dotnet.microsoft.com/en-us/download/dotnet/7.0

   # 2. 安装 nncase 基础包（在线）
   pip install nncase==2.9.0

   # 3. 下载 nncase-kpu 离线 wheel（nncase 和 nncase-kpu 版本必须一致）
   #    https://github.com/kendryte/nncase/releases
   #    例如：nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl

   # 4. 离线安装 nncase-kpu
   pip install nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl
   ```
   > **注意**：Windows 下 `nncase-kpu` 没有发布到 PyPI，必须从 GitHub Releases 下载对应版本的 `.whl` 文件离线安装。nncase 和 nncase-kpu 版本号必须一致。

## 命令模板

### 方式一：一键转换脚本（推荐）

```powershell
cd K230_Yolov5n
.\convert.ps1 -Model ..\yolov5-7.0\best.onnx -Dataset <校准图片目录>
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-Model` | ONNX 模型路径 | 必填 |
| `-Dataset` | 校准图片目录（20张） | 必填 |
| `-InputWidth` | 输入宽度 | `320` |
| `-InputHeight` | 输入高度 | `320` |
| `-PTQOption` | 量化选项 | `0` |
| `-SkipVerify` | 跳过精度验证 | 不跳过 |

脚本自动执行三步：
1. ONNX → kmodel 量化编译
2. 生成测试输入 bin 文件
3. ONNX vs kmodel 余弦相似度验证

### 方式二：分步手动执行

#### Step 1: ONNX → kmodel

```bash
cd K230_Yolov5n/tools
python to_kmodel.py \
  --target k230 \
  --model <ONNX路径> \
  --dataset <校准集目录> \
  --input_width 320 \
  --input_height 320 \
  --ptq_option 0
```

输出：`<模型名>.kmodel`（与 ONNX 同目录）

#### Step 2: 生成测试输入

```bash
python save_bin.py \
  --image <校准集任意一张图> \
  --save_path . \
  --input_width 320 \
  --input_height 320
```

输出：`onnx_input_float32.bin` 和 `kmodel_input_uint8.bin`

#### Step 3: 精度验证（余弦相似度）

```bash
python simulate.py \
  --model <ONNX路径> \
  --kmodel <kmodel路径> \
  --model_input onnx_input_float32.bin \
  --kmodel_input kmodel_input_uint8.bin \
  --input_width 320 \
  --input_height 320
```

输出每个输出层的余弦相似度（越接近 1 越好，通常 > 0.95 可用）。

#### Step 4: kmodel 推理测试（可选）

修改 `test_det_kmodel.py` 中的路径后直接运行：

```bash
python test_det_kmodel.py
```

输出 `kmodel_det_result.jpg` 可视化检测结果。

## 量化参数 (PTQOption)

| 值 | 说明 | 适用场景 |
|----|------|----------|
| `0` | uint8 量化（默认） | 通用场景，体积最小 |
| `1` | 权重 int16，激活 uint8 | 精度要求高 |
| `2` | 全 int16 量化 | 最高精度，体积更大 |

## 精度验证结果解读

- **cosine > 0.98**：精度损失极小，放心部署
- **cosine 0.95~0.98**：轻微损失，通常不影响检测效果
- **cosine 0.90~0.95**：有明显损失，建议换 PTQOption 重试
- **cosine < 0.90**：严重损失，检查校准集质量和输入尺寸匹配

## 转换参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--target` | 目标芯片 | `k230` |
| `--model` | ONNX 模型路径 | 必填 |
| `--dataset` | 校准集目录 | 训练集抽 20 张 |
| `--input_width` | 输入宽度 | 与训练一致（如 320） |
| `--input_height` | 输入高度 | 与训练一致（如 320） |
| `--ptq_option` | 量化选项 | `0` |

> **注意**：输入尺寸会自动向上取整为 32 的倍数。例如 300→320，310→320。

## K230 部署

转换完成后：

1. 将 `.kmodel` 复制到 K230 设备：
   ```
   设备路径: /sdcard/mp_deployment_source/
   ```

2. 修改 `K230_Yolov5n/mp_deployment_source/deploy_config.json`：
   ```json
   {
     "kmodel_path": "/sdcard/mp_deployment_source/<你的模型>.kmodel",
     ...
   }
   ```

3. 运行推理：参考 `K230_Yolov5n/main.py` 或 `mp_deployment_source/main.py`

## 完整部署流程

```
yolo-train → yolo-export → yolo-kmodel → K230 部署
  训练         导出ONNX      转kmodel      端侧运行
```

参考：
- 训练：`yolo-train` skill
- 导出 ONNX：`yolo-export` skill
- 部署源码：`K230_Yolov5n/mp_deployment_source/`
- K230 端推理：`K230_Yolov5n/main.py`

## 常见问题

1. **nncase 导入失败**：
   - 确认 `nncase` **和** `nncase-kpu` 都已安装（两个是独立包）
   - 确认 nncase 与 nncase-kpu **版本号一致**（如都是 2.9.0）
   - Windows 下 `nncase-kpu` 需从 [GitHub Releases](https://github.com/kendryte/nncase/releases) 下载离线 wheel
   - 确认 dotnet-7 runtime 已安装且加入 PATH 环境变量
2. **校准图片不足**：`to_kmodel.py` 需要至少 20 张校准图片
3. **ONNX 加载失败**：检查 opset 版本是否为 12，尝试重新导出 ONNX
4. **精度损失严重**：增大校准集（多抽几张），尝试 PTQOption 1 或 2
5. **kmodel 体积过大**：使用默认 uint8 量化 (PTQOption=0)，体积约为 ONNX 的 1/4
6. **simulate 余弦相似度低**：检查 `input_width/height` 是否与训练时一致
