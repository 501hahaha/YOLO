# K230 kmodel 转换与测试工具

## 环境准备

### Linux (K230 开发板 / Ubuntu)

```bash
# 1. 安装 dotnet-7 (nncase 2.x 依赖)
sudo apt-get install -y dotnet-sdk-7.0

# 2. 在线安装 nncase + nncase-kpu
pip install --upgrade pip
pip install nncase==2.9.0
pip install nncase-kpu==2.9.0

# 3. 安装其他依赖
pip install onnx onnxruntime onnxsim pillow numpy
```

### Windows

```powershell
# 1. 安装 dotnet-7
#    从 https://dotnet.microsoft.com/download/dotnet/7.0 下载安装
#    安装后确保 dotnet 在 PATH 环境变量中

# 2. 在线安装 nncase
pip install nncase==2.9.0

# 3. 离线安装 nncase-kpu (Windows 不支持 pip 在线安装)
#    从 https://github.com/kendryte/nncase/releases 下载对应版本 .whl
#    本目录已附带 nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl
pip install nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl

# 4. 安装其他依赖
pip install onnx onnxruntime onnxsim pillow numpy
```

## 完整流程: .pt → ONNX → kmodel → 部署

### 第一步: 导出 ONNX

```bash
cd ../yolov5-7.0
python export.py --weights best.pt --imgsz 320 --batch 1 --include onnx --opset 12
```

### 第二步: 转换 kmodel

```bash
cd ../K230_Yolov5n/tools
python to_kmodel.py --target k230 --model ../../yolov5-7.0/runs/train/ball_320/weights/best.onnx --dataset <校准图片目录> --input_width 320 --input_height 320 --ptq_option 0
```

| 参数 | 说明 |
|------|------|
| `--target` | 目标芯片: `k230` |
| `--model` | 输入的 ONNX 模型路径 |
| `--dataset` | 校准图片目录 (20张即可) |
| `--input_width` | 输入宽度 (与训练一致) |
| `--input_height` | 输入高度 (与训练一致) |
| `--ptq_option` | 量化选项: 0=uint8, 1=NoClip+int16权重, 2=全int16 |

### 第三步: 精度验证 (可选)

```bash
# ONNX 推理测试
python test_det_onnx.py --model best.onnx --image test.jpg

# kmodel 推理测试 (PC端模拟)
python test_det_kmodel.py --kmodel best.kmodel --image test.jpg

# kmodel vs ONNX 余弦相似度对比
python simulate.py --model best.onnx --kmodel best.kmodel --model_input onnx_input.bin --kmodel_input kmodel_input.bin
```

### 第四步: 部署到 K230

```bash
# 将以下文件拷贝到 K230 的 /sdcard/mp_deployment_source/
#  - best.kmodel
#  - deploy_config.json (修改 kmodel_path 为你的 kmodel 文件名)
#  - main.py
```

### K230 上首次运行

```bash
# K230 上运行 kmodel 推理
python3 main.py
```

## deploy_config.json 配置说明

```json
{
    "kmodel_path": "best.kmodel",          // kmodel 文件名
    "categories": ["class_name"],          // 类别名列表
    "num_classes": 1,                       // 类别数
    "img_size": [320, 320],                // 输入尺寸
    "confidence_threshold": 0.85,           // 置信度阈值
    "nms_threshold": 0.45,                 // NMS 阈值
    "nms_option": false,                   // NMS 开关
    "model_type": "AnchorBaseDet",         // 模型类型
    "anchors": [[10,13,16,30,33,23], [30,61,62,45,59,119], [116,90,156,198,373,326]]
}
```

> anchors 需要从 `yolov5-7.0/models/yolov5n.yaml` 中的 anchors 复制过来
