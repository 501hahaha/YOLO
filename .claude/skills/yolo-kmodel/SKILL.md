---
name: yolo-kmodel
description: YOLOv5 ONNX 到 K230 kmodel 的转换、nncase 量化、精度验证和部署配置准备。用于完成 YOLOv5 检测模型的 K230 转换；普通 YOLO 推理不触发此 skill。
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---


## 项目约束

- 先确认训练进程已经退出，并确认 `yolov5-7.0/runs/train/ball_320/weights/best.pt` 存在；不要在训练中的 checkpoint 上转换。
- 使用项目环境 `.conda/yolov5_train_py310/python.exe`，不要把依赖安装到系统 Python。
- 转换工具以 `K230_Yolov5n/tools/to_kmodel.py` 为准；不要覆盖 `K230_Yolov5n/mp_deployment_source/best.kmodel` 或已有配置。
- 新模型输出优先放在本次训练目录 `yolov5-7.0/runs/train/ball_320/weights/`，部署配置单独放在该训练目录下。

## 固定转换契约

- YOLOv5n 检测模型，类别为 `ball`，输入 `320×320`，导出 batch 为 `1`。
- 导出 ONNX 使用静态 shape、opset `12`；不要添加 `--dynamic`。
- 本地脚本的 nncase 组合是 `nncase==2.9.0` 与随项目提供的 `nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl`。项目机器已经有 dotnet `7.0.410`。
- 当前 PyTorch 为 2.11，导出依赖固定为 `onnxscript==0.6.2`、`onnx_ir==1.0.0`、`onnx==1.17.0`、`onnxruntime==1.19.0`、`onnxsim==0.4.36`、`numpy==1.26.4`、`scipy==1.14.1`、`ml_dtypes==0.5.1`。
- Windows nncase 2.9.0 还需要 LLVM OpenMP runtime；本机项目环境中应存在 `libomp140.x86_64.dll`，缺失时先修复 DLL 依赖再转换。
- `to_kmodel.py` 的编译参数是 `target=k230`、`input_shape=[1,3,320,320]`、`input_layout=NCHW`、`input_type=uint8`、`input_range=[0,1]`、`swapRB=False`、`mean=[0,0,0]`、`std=[1,1,1]`、`quant_type=uint8`。
- `PTQOption=0` 是默认方案，校准样本数为 20；校准目录只放图片，不放 YOLO `.txt` 标注。

## 执行顺序

若训练在后台运行且用户要求训练结束后继续处理，可调用 `scripts/k230/finish_training_and_convert.ps1 -TrainingPid <PID> -Shutdown`。该脚本必须等待训练进程退出，并且只在导出、转换、simulator 验证、部署配置和产物检查全部成功后关机；失败时保留 `runs/train/ball_320/k230_pipeline.log`，不得关机。

1. 导出 ONNX：

   ```powershell
   $ProjectRoot = (Get-Location).Path
   $py = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
   $pt = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.pt'
   & $py (Join-Path $ProjectRoot 'yolov5-7.0\export.py') --weights $pt --imgsz 320 --batch-size 1 --include onnx --opset 12 --simplify --device cpu
   ```

2. 在同一环境安装转换依赖。Windows 的 KPU wheel 必须和 nncase 版本一致；`export.py` 已为 PyTorch 2.11 显式切换到 legacy ONNX exporter，确保 opset 12 真正生效：

   ```powershell
   & $py -m pip install 'nncase==2.9.0' 'numpy==1.26.4' 'scipy==1.14.1' 'onnx==1.17.0' 'onnxruntime==1.19.0' 'onnxsim==0.4.36' 'onnxscript==0.6.2' 'onnx_ir==1.0.0' 'ml_dtypes==0.5.1'
   & $py -m pip install --no-deps (Join-Path $ProjectRoot 'K230_Yolov5n\nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl')
   ```

3. 转换：

   ```powershell
   $ProjectRoot = (Get-Location).Path
   $onnx = Join-Path $ProjectRoot 'yolov5-7.0\runs\train\ball_320\weights\best.onnx'
   $calib = Join-Path $ProjectRoot 'ball_image\datasets\yolo_dataset_10000\images\val'
   Push-Location (Join-Path $ProjectRoot 'K230_Yolov5n\tools')
   & $py '.\to_kmodel.py' --target k230 --model $onnx --dataset $calib --input_width 320 --input_height 320 --ptq_option 0
   Pop-Location
   ```

4. 验证 `best.kmodel` 非空。当前本地 nncase `Simulator` 无法直接加载 `target=k230` 产物，因此使用相同 ONNX、输入和 PTQ 参数额外生成 `target=cpu` 参考 kmodel，再用 `save_bin.py` 与 `simulate.py` 比较输出并记录 cosine similarity；这不是 K230 板端验证。数值明显偏低时，先检查输入尺寸、RGB/NCHW、校准图和 opset，再尝试 PTQOption `1` 或 `2`。

5. 生成部署配置时，将 `categories` 设置为 `["ball"]`，`num_classes` 设置为 `1`，`img_size` 设置为 `[320,320]`，`nms_threshold` 默认 `0.45`，`nms_option` 默认 `false`。`anchors` 必须从最终 checkpoint/模型实际使用的 anchors 取得，不能直接复制参考目录里属于其他模型的自定义 anchors。

## 交付检查

- `best.pt`、`best.onnx`、`best.kmodel` 均存在且大小大于 0。
- `deploy_config.json` 的模型文件名、类别、输入尺寸、anchors、nncase 版本与新模型一致。
- 保留校准目录、转换脚本版本和验证输出，便于重新生成；官方说明指出 kmodel 本身不携带 nncase 版本。
- PC simulator 通过不等于板端已验证。只有在 K230 上加载并运行后，才能报告板端推理成功。

详细项目路径和命令见 `docs/YOLO_K230_WORKFLOW.md`。官方参考：`https://www.kendryte.com/k230_canmv/zh/main/example/ai/yolo_battle.html`。

## 安全边界

除非用户明确要求，不烧录固件、不上传模型到开发板、不删除旧模型。用户明确要求关机时，只能在训练、转换、验证和工程整理全部完成并做完最后状态检查后执行关机。
