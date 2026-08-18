$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPath = [IO.Path]::GetFullPath((Join-Path $projectRoot ".venv"))
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & (Join-Path $PSScriptRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Base setup failed." }
}

& $venvPython -m pip uninstall -y paddlepaddle paddlepaddle-gpu
if ($LASTEXITCODE -ne 0) { throw "Failed to remove an existing PaddlePaddle build." }
& $venvPython -m pip install paddlepaddle-gpu==3.2.2 `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddlePaddle GPU (CUDA 12.6)." }
& $venvPython -c "import paddle; print('Paddle:', paddle.__version__); print('Device:', paddle.device.get_device()); print('CUDA:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count()); paddle.utils.run_check()"
if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle GPU verification failed." }

Write-Host "Setup complete. Start with: .\.venv\Scripts\python.exe app.py"
