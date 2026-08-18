$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The project environment does not exist. Run setup.ps1 first."
}

& $venvPython -m pip uninstall -y paddlepaddle-gpu
if ($LASTEXITCODE -ne 0) { throw "Failed to uninstall the CUDA 13.0 build." }
& $venvPython -m pip install --no-cache-dir paddlepaddle-gpu==3.2.2 `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
if ($LASTEXITCODE -ne 0) { throw "Failed to install the CUDA 12.6 build." }
& $venvPython -c "import paddle; print('Paddle:', paddle.__version__); print('Device:', paddle.device.get_device()); print('CUDA:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count()); paddle.utils.run_check()"
if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle GPU verification failed." }

Write-Host "GPU repair complete. Start with: .\.venv\Scripts\python.exe app.py"
