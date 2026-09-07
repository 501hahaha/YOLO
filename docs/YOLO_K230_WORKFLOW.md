# 小球 YOLOv5 到 K230 工程流程

本文档记录本项目当前可复现的训练、导出和 K230 `kmodel` 转换流程。转换参数以项目内的 `K230_Yolov5n` 为准，并参考嘉楠官方的《K230 YOLO 大作战》文档。

## 工程布局

| 路径 | 用途 |
|---|---|
| `ball_image/datasets/yolo_dataset_10000/` | 10,000 张图片及 YOLO 标注，`train/val/test = 8000/1000/1000` |
| `yolov5-7.0/` | YOLOv5 v7.0 训练、验证和 ONNX 导出 |
| `K230_Yolov5n/` | K230 转换脚本、部署示例和本地参考参数 |
| `.conda/yolov5_train_py310/` | 本机训练/转换共用的 Conda 环境 |
| `yolov5-7.0/runs/train/ball_320/` | 正式训练输出；包含 `weights/best.pt` 和训练指标 |
| `.codex/skills/` | Codex 项目级 YOLO skills |

不要覆盖 `K230_Yolov5n/mp_deployment_source/` 中已有的 `best.kmodel`。该目录里的模型和 `deploy_config.json` 可能属于此前的工程；新模型先输出到本次训练目录下，再单独生成部署配置。

## 固定参数

- 模型：YOLOv5n，单类别 `ball`。
- 训练/导出/转换输入：`320×320`。
- K230 导出 batch：`1`，ONNX opset：`12`，不使用 dynamic shape。
- K230 转换：`target=k230`，`input_shape=[1,3,320,320]`，`input_layout=NCHW`。
- 转换前处理：`input_type=uint8`、`input_range=[0,1]`、`swapRB=False`、`mean=[0,0,0]`、`std=[1,1,1]`。
- 默认量化：`PTQOption=0`，即本地脚本的 uint8 数据/uint8 权重方案；校准样本 20 张。
- 校准目录只放图片，不放 `.txt` 标注。当前可直接使用 `ball_image/datasets/yolo_dataset_10000/images/val/`，本地脚本会取其中前 20 张。

## 训练

正式训练使用项目内环境和本机 GPU，命令如下（已启动的任务输出到 `runs/train/ball_320`）：

```powershell
$ProjectRoot = (Get-Location).Path
$py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
& (Join-Path $ProjectRoot 'yolov5-7.0\train.py') `
  --data (Join-Path $ProjectRoot 'ball_image\datasets\yolo_dataset_10000\data.yaml') `
  --weights (Join-Path $ProjectRoot 'yolov5-7.0\yolov5n.pt') `
  --img 320 --batch-size 128 --epochs 100 --workers 2 --device 0 `
  --project (Join-Path $ProjectRoot 'yolov5-7.0\runs\train') `
  --name ball_320 --exist-ok --patience 30 --noplots
```

训练完成的必要条件是训练进程退出且以下两个文件存在：

```text
yolov5-7.0/runs/train/ball_320/weights/best.pt
yolov5-7.0/runs/train/ball_320/results.csv
```

## `.pt -> .onnx -> .kmodel`

如果训练仍在后台运行，可使用 `scripts/k230/finish_training_and_convert.ps1` 等待指定训练 PID。该脚本会在训练结束后自动安装转换依赖、导出、转换、验证和生成部署配置；只有传入 `-Shutdown` 且所有检查通过时才执行关机。运行前应先确认 PID 是本次 `ball_320` 训练进程。

转换依赖只安装到项目 Conda 环境，不修改系统 Python。Windows 需要 `dotnet 7` 和项目内的离线 `nncase_kpu` wheel。当前项目使用 PyTorch 2.11，因此 ONNX 导出栈固定为 `onnxscript==0.6.2`、`onnx_ir==1.0.0`、`onnx==1.17.0`；这组版本支持当前导出器，同时生成的静态 opset 12 模型可被本地 nncase 2.9.0 导入：

Windows 的 nncase 2.9.0 还需要 LLVM OpenMP 运行库；本机文件为 `.conda/yolov5_train_py310/Lib/site-packages/libomp140.x86_64.dll`。新环境若导入 `nncase` 报 DLL 缺失，应安装对应的 LLVM OpenMP runtime，不要用其他 OpenMP DLL 改名替代。

```powershell
$ProjectRoot = (Get-Location).Path
$py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
& $py -m pip install 'nncase==2.9.0' 'numpy==1.26.4' 'scipy==1.14.1' 'onnx==1.17.0' 'onnxruntime==1.19.0' 'onnxsim==0.4.36' 'onnxscript==0.6.2' 'onnx_ir==1.0.0' 'ml_dtypes==0.5.1'
& $py -m pip install --no-deps (Join-Path $ProjectRoot 'K230_Yolov5n\nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl')
```

PyTorch 2.6 及以上默认使用 dynamo ONNX 导出器，而 nncase 2.9.0 不接受其常见的 opset 18/`allowzero` 结构。本项目的 `yolov5-7.0/export.py` 已在检测到该参数时显式使用 legacy exporter，使 `--opset 12` 真正生效；不要删除这一兼容处理。

训练结束后导出静态 ONNX：

```powershell
$ProjectRoot = (Get-Location).Path
$py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
$pt = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.pt'
& $py (Join-Path $ProjectRoot 'yolov5-7.0\export.py') `
  --weights $pt --imgsz 320 --batch-size 1 --include onnx `
  --opset 12 --simplify --device cpu
```

再使用项目参考脚本转换。输出的 `best.kmodel` 与 `best.onnx` 同目录：

```powershell
$ProjectRoot = (Get-Location).Path
$py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
$onnx = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.onnx'
$calib = Join-Path $ProjectRoot 'ball_image\datasets\yolo_dataset_10000\images\val'
Push-Location (Join-Path $ProjectRoot 'K230_Yolov5n\tools')
& $py '.\to_kmodel.py' --target k230 --model $onnx --dataset $calib `
  --input_width 320 --input_height 320 --ptq_option 0
Pop-Location
```

## 转换验证

使用同一张校准图生成 ONNX 和输入。由于本地 nncase 2.9.0 的 `Simulator` 无法直接加载 `target=k230` 产物，验证脚本会用相同 ONNX、输入和 PTQ 参数额外生成一个 `target=cpu` 参考 kmodel，再用它运行 `simulate.py` 做数值相似度检查；这一步不是 K230 板端验证：

```powershell
$ProjectRoot = (Get-Location).Path
$py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
$calib = Join-Path $ProjectRoot 'ball_image\datasets\yolo_dataset_10000\images\val'
$img = Get-ChildItem $calib -File |
  Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png','.bmp') } |
  Sort-Object Name | Select-Object -First 1 -ExpandProperty FullName
$out = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\k230_verify'
$onnx = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.onnx'
$kmodel = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.kmodel'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location (Join-Path $ProjectRoot 'K230_Yolov5n\tools')
& $py '.\save_bin.py' --image $img --save_path $out --input_width 320 --input_height 320
$cpu_dir = Join-Path $out 'cpu_reference'
New-Item -ItemType Directory -Force -Path $cpu_dir | Out-Null
$cpu_onnx = Join-Path $cpu_dir 'best.onnx'
Copy-Item $onnx $cpu_onnx -Force
& $py '.\to_kmodel.py' --target cpu --model $cpu_onnx --dataset (Split-Path $img) --input_width 320 --input_height 320 --ptq_option 0
& $py '.\simulate.py' --model $onnx --kmodel (Join-Path $cpu_dir 'best.kmodel') `
  --model_input (Join-Path $out 'onnx_input_float32.bin') `
  --kmodel_input (Join-Path $out 'kmodel_input_uint8.bin') `
  --input_width 320 --input_height 320
Pop-Location
```

应记录每个输出层的 cosine similarity，并且至少检查一张图片的 ONNX 与 Kmodel 检测框。余弦相似度只能证明 PC simulator 的数值接近，不能替代真实 K230 板端验证。

## 部署配置

K230 端使用 `AnchorBaseDet`。`anchors` 必须取训练完成的 checkpoint/模型实际使用的 anchors，不能直接照搬现有参考目录里属于其他模型的自定义 anchors。新配置至少包含：

```json
{
  "chip_type": "k230",
  "nncase_version": "2.9.0",
  "model_type": "AnchorBaseDet",
  "img_size": [320, 320],
  "categories": ["ball"],
  "num_classes": 1,
  "confidence_threshold": 0.5,
  "nms_threshold": 0.45,
  "nms_option": false,
  "kmodel_path": "ball_yolov5n_320.kmodel"
}
```

把生成的 kmodel、该配置和 `K230_Yolov5n/mp_deployment_source/main.py` 复制到开发板的 `/sdcard/mp_deployment_source/` 后再运行。板端固件的 nncase runtime 版本要与转换版本匹配；如果不匹配，应先按官方版本对应关系处理。

## 官方参考

- [K230 YOLO 大作战](https://www.kendryte.com/k230_canmv/zh/main/example/ai/yolo_battle.html)：YOLOv5 检测训练、ONNX 导出、`to_kmodel.py` 参数和部署流程。
- [K230 YOLO 模块 API 手册](https://github.com/kendryte/k230_canmv_docs/blob/main/zh/api/aidemo/YOLO%20%E6%A8%A1%E5%9D%97%20API%20手册.md)：K230 端 YOLOv5 检测参数和输入尺寸。
- [nncase 官方仓库](https://github.com/kendryte/nncase)：K230 编译器与版本发布信息。
- [K230 nncase runtime 版本说明](https://github.com/kendryte/k230_docs/blob/main/en/03_other/K230_SDK_Updating_nncase_Runtime_Library_Guide.md)：kmodel 不携带 nncase 版本，编译环境和板端 runtime 需要自行记录并匹配。
