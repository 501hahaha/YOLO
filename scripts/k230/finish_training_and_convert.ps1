# 等待正式训练结束后自动完成 YOLOv5 -> K230 kmodel 流程。
# 默认只转换和验证；只有显式传入 -Shutdown 才会关机。
[CmdletBinding()]
param(
    [int]$TrainingPid = 86232,
    [switch]$Shutdown
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.conda\yolov5_train_py310\python.exe'
$SitePackages = Join-Path $ProjectRoot '.conda\yolov5_train_py310\Lib\site-packages'
$OpenMpDll = Join-Path $SitePackages 'libomp140.x86_64.dll'
$YoloRoot = Join-Path $ProjectRoot 'yolov5-7.0'
$RunDir = Join-Path $YoloRoot 'runs\train\ball_320'
$WeightsDir = Join-Path $RunDir 'weights'
$BestPt = Join-Path $WeightsDir 'best.pt'
$Onnx = Join-Path $WeightsDir 'best.onnx'
$Kmodel = Join-Path $WeightsDir 'best.kmodel'
$ToolsDir = Join-Path $ProjectRoot 'K230_Yolov5n\tools'
$Wheel = Join-Path $ProjectRoot 'K230_Yolov5n\nncase_kpu-2.9.0-py2.py3-none-win_amd64.whl'
$CalibrationDir = Join-Path $ProjectRoot 'ball_image\datasets\yolo_dataset_10000\images\val'
$SampleImage = $null
$VerifyDir = Join-Path $RunDir 'k230_verify'
$CpuVerifyDir = Join-Path $VerifyDir 'cpu_reference'
$DeployDir = Join-Path $RunDir 'k230_deployment'
$ConfigScript = Join-Path $ProjectRoot 'scripts\k230\generate_k230_config.py'
$LogPath = Join-Path $RunDir 'k230_pipeline.log'

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Log ("RUN " + $FilePath + " " + ($Arguments -join ' '))
    $tempStdout = Join-Path $env:TEMP ("yolo_k230_" + [Guid]::NewGuid().ToString('N') + '.stdout.log')
    $tempStderr = Join-Path $env:TEMP ("yolo_k230_" + [Guid]::NewGuid().ToString('N') + '.stderr.log')
    try {
        # 用独立进程捕获 stdout/stderr，避免 Windows PowerShell 把 Python
        # 的普通 warning 当成 ErrorRecord 传给外层 Stop 异常处理。
        $argumentString = ''
        foreach ($argument in $Arguments) {
            $argumentText = if ($null -eq $argument) { '' } else { [Convert]::ToString($argument) }
            if ($argumentString.Length -gt 0) { $argumentString += ' ' }
            $argumentString += ('"{0}"' -f $argumentText.Replace('"', '\"'))
        }
        if ([string]::IsNullOrWhiteSpace($argumentString)) { throw "Empty argument list: $FilePath" }
        $process = Start-Process -FilePath $FilePath -ArgumentList $argumentString -WorkingDirectory (Get-Location).Path `
            -WindowStyle Hidden -RedirectStandardOutput $tempStdout -RedirectStandardError $tempStderr -Wait -PassThru
        $exitCode = $process.ExitCode
        $stdout = if (Test-Path -LiteralPath $tempStdout) { Get-Content -LiteralPath $tempStdout -Raw -ErrorAction SilentlyContinue } else { '' }
        $stderr = if (Test-Path -LiteralPath $tempStderr) { Get-Content -LiteralPath $tempStderr -Raw -ErrorAction SilentlyContinue } else { '' }
        $combined = (($stdout, $stderr) -join "`n").Trim()
        if ($combined) { Add-Content -LiteralPath $LogPath -Value $combined -Encoding UTF8 }
        if ($exitCode -ne 0) { throw "Command failed, exit code=$exitCode : $FilePath" }
        return $combined
    } finally {
        if (Test-Path -LiteralPath $tempStdout) { Remove-Item -LiteralPath $tempStdout -Force }
        if (Test-Path -LiteralPath $tempStderr) { Remove-Item -LiteralPath $tempStderr -Force }
    }
}

try {
    Write-Log 'PIPELINE_START'
    if (-not (Test-Path -LiteralPath $Python)) { throw "Project Python not found: $Python" }
    if (-not (Test-Path -LiteralPath $Wheel)) { throw "nncase-kpu wheel not found: $Wheel" }
    if (-not (Test-Path -LiteralPath $OpenMpDll)) { throw "nncase Windows prerequisite missing: $OpenMpDll (install LLVM OpenMP runtime)" }

    if ($TrainingPid -gt 0) {
        Write-Log "Waiting for training PID=$TrainingPid to exit"
        while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
            Start-Sleep -Seconds 30
        }
    }
    Start-Sleep -Seconds 5

    $remaining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*yolov5-7.0*train.py*ball_320*' }
    if ($remaining) { throw 'ball_320 training process still exists; conversion stopped' }
    if (-not (Test-Path -LiteralPath $BestPt)) { throw "Training did not produce best.pt: $BestPt" }
    $ResultsCsv = Join-Path $RunDir 'results.csv'
    if (-not (Test-Path -LiteralPath $ResultsCsv)) { throw 'Training results.csv not found' }
    $SampleImage = Get-ChildItem -LiteralPath $CalibrationDir -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg', '.jpeg', '.png', '.bmp') } |
        Sort-Object Name |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $SampleImage) { throw "No calibration image found: $CalibrationDir" }
    Write-Log ('SIMULATION_SAMPLE ' + $SampleImage)
    Write-Log ((Get-Item -LiteralPath $BestPt).FullName + ' size=' + (Get-Item -LiteralPath $BestPt).Length)

    Invoke-External $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', 'nncase==2.9.0', 'numpy==1.26.4', 'scipy==1.14.1', 'onnx==1.17.0', 'onnxruntime==1.19.0', 'onnxsim==0.4.36', 'onnxscript==0.6.2', 'onnx_ir==1.0.0', 'ml_dtypes==0.5.1') | Out-Null
    Invoke-External $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-warn-script-location', '--no-deps', $Wheel) | Out-Null
    Invoke-External $Python @('-c', 'import nncase, onnx, onnxruntime, onnxsim, onnxscript, onnx_ir; print(''nncase='' + getattr(nncase, ''__version__'', ''unknown'')); print(''onnx='' + onnx.__version__); print(''onnxruntime='' + onnxruntime.__version__); print(''onnxsim='' + getattr(onnxsim, ''__version__'', ''unknown'')); print(''onnxscript='' + onnxscript.__version__); print(''onnx_ir='' + getattr(onnx_ir, ''__version__'', ''unknown''))') | Out-Null

    Invoke-External $Python @((Join-Path $YoloRoot 'export.py'), '--weights', $BestPt, '--imgsz', '320', '--batch-size', '1', '--include', 'onnx', '--opset', '12', '--simplify', '--device', 'cpu') | Out-Null
    if (-not (Test-Path -LiteralPath $Onnx)) { throw "ONNX export failed: $Onnx" }
    Invoke-External $Python @('-c', 'import sys, onnx; m=onnx.load(sys.argv[1]); onnx.checker.check_model(m); assert m.opset_import[0].version == 12, m.opset_import[0].version; assert [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim] == [1,3,320,320]; print(''onnx_contract=ok'')', $Onnx) | Out-Null

    Push-Location $ToolsDir
    try {
        Invoke-External $Python @('.\to_kmodel.py', '--target', 'k230', '--model', $Onnx, '--dataset', $CalibrationDir, '--input_width', '320', '--input_height', '320', '--ptq_option', '0') | Out-Null
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $Kmodel)) { throw "kmodel conversion failed: $Kmodel" }

    New-Item -ItemType Directory -Force -Path $VerifyDir,$CpuVerifyDir,$DeployDir | Out-Null
    Invoke-External $Python @((Join-Path $ToolsDir 'save_bin.py'), '--image', $SampleImage, '--save_path', $VerifyDir, '--input_width', '320', '--input_height', '320') | Out-Null

    # K230 target 生成的 kmodel 由板端 runtime 执行；本地 nncase Simulator
    # 对 target=k230 文件不提供加载支持，因此用同一 ONNX/量化参数生成
    # target=cpu 的参考 kmodel 做数值相似度检查，不能把它当成板端验证。
    $CpuVerifyDir = [System.IO.Path]::Combine($RunDir, 'k230_verify', 'cpu_reference')
    $CpuOnnxPath = [System.IO.Path]::Combine($CpuVerifyDir, 'best.onnx')
    $CpuKmodelPath = [System.IO.Path]::Combine($CpuVerifyDir, 'best.kmodel')
    Write-Log ('CPU_REFERENCE_PATH ' + $CpuOnnxPath)
    Copy-Item -LiteralPath $Onnx -Destination $CpuOnnxPath -Force
    Push-Location $ToolsDir
    try {
        $cpuArguments = [string[]]@('.\to_kmodel.py', '--target', 'cpu', '--model', [string]$CpuOnnxPath, '--dataset', [string]$CalibrationDir, '--input_width', '320', '--input_height', '320', '--ptq_option', '0')
        Invoke-External -FilePath $Python -Arguments $cpuArguments | Out-Null
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $CpuKmodelPath)) { throw "CPU reference kmodel conversion failed: $CpuKmodelPath" }

    Push-Location $ToolsDir
    try {
        $simOutput = Invoke-External $Python @('.\simulate.py', '--model', $Onnx, '--kmodel', $CpuKmodelPath, '--model_input', (Join-Path $VerifyDir 'onnx_input_float32.bin'), '--kmodel_input', (Join-Path $VerifyDir 'kmodel_input_uint8.bin'), '--input_width', '320', '--input_height', '320')
    } finally {
        Pop-Location
    }
    $matches = [regex]::Matches($simOutput, 'cosine similarity\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
    if ($matches.Count -eq 0) { throw 'simulate produced no cosine similarity' }
    foreach ($match in $matches) {
        $cosine = [double]$match.Groups[1].Value
        if ($cosine -lt 0.95) { throw "cosine similarity too low: $cosine" }
    }

    Copy-Item -LiteralPath $Kmodel -Destination (Join-Path $DeployDir 'ball_yolov5n_320.kmodel') -Force
    Invoke-External $Python @($ConfigScript, '--weights', $BestPt, '--output', (Join-Path $DeployDir 'deploy_config.json'), '--model-name', 'ball_yolov5n_320.kmodel', '--class-name', 'ball', '--confidence', '0.5', '--nms', '0.45', '--nncase-version', '2.9.0') | Out-Null

    foreach ($artifact in @($BestPt, $Onnx, $Kmodel, $CpuKmodelPath, (Join-Path $DeployDir 'ball_yolov5n_320.kmodel'), (Join-Path $DeployDir 'deploy_config.json'))) {
        $item = Get-Item -LiteralPath $artifact -ErrorAction Stop
        if ($item.Length -le 0) { throw "Artifact is empty: $artifact" }
        Write-Log ("ARTIFACT " + $item.FullName + " size=" + $item.Length)
    }
    Write-Log 'CPU_REFERENCE_SIMULATE_PASS cosine>=0.95; K230 target requires board runtime validation'
    Write-Log 'PIPELINE_SUCCESS'

    if ($Shutdown) {
        Write-Log 'SHUTDOWN_REQUESTED'
        & shutdown.exe /s /t 0
    }
} catch {
    Write-Log ('PIPELINE_FAILED ' + $_.Exception.Message)
    Write-Log ('PIPELINE_FAILED_AT ' + $_.InvocationInfo.PositionMessage)
    Write-Log ('PIPELINE_FAILED_STACK ' + $_.ScriptStackTrace)
    exit 1
}
