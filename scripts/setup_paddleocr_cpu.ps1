$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The project environment does not exist. Run scripts\setup.ps1 first."
}

# The CPU and GPU Paddle packages expose the same `paddle` module and must not
# coexist. This intentionally switches the environment to the CPU runtime.
& $venvPython -m pip uninstall -y paddlepaddle paddlepaddle-gpu
if ($LASTEXITCODE -ne 0) { throw "Failed to remove the existing PaddlePaddle build." }

& $venvPython -m pip install paddlepaddle==3.2.2
if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddlePaddle CPU." }

# PaddleX imports ModelScope, which imports PyTorch even though these PaddleOCR
# models do not use it. Preserve a working PyTorch install, but repair a broken
# CUDA build with the official CPU wheels so PaddleOCR can import reliably.
& $venvPython -c "import torch; print('PyTorch dependency:', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "The installed PyTorch runtime cannot load; replacing it with CPU wheels."
    & $venvPython -m pip uninstall -y torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "Failed to remove the broken PyTorch build." }
    & $venvPython -m pip install torch==2.12.1 torchvision==0.27.1 `
        --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch." }
}

& $venvPython -c "from paddleocr import TextDetection, TextRecognition; import paddle; print('Paddle:', paddle.__version__); print('Device:', paddle.device.get_device()); print('CUDA:', paddle.is_compiled_with_cuda()); assert not paddle.is_compiled_with_cuda(), 'Expected the CPU PaddlePaddle build'; paddle.utils.run_check(); print('PaddleOCR CPU imports passed.')"
if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle CPU verification failed." }

Write-Host "PaddleOCR CPU setup complete. Select 'PaddleOCR CPU' in the app."
