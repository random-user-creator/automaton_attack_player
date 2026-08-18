$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The project environment does not exist. Run setup.ps1 first."
}

# ONNX Runtime documents that only one of its CPU/GPU Python packages should
# be installed in an environment. Versions 1.21-1.26 use CUDA 12.x + cuDNN 9.x;
# 1.27 and newer PyPI builds use CUDA 13.x.
& $venvPython -m pip uninstall -y onnxruntime onnxruntime-gpu
if ($LASTEXITCODE -ne 0) { throw "Failed to remove the existing ONNX Runtime." }

& $venvPython -m pip install "onnxruntime-gpu>=1.21,<1.27"
if ($LASTEXITCODE -ne 0) { throw "Failed to install ONNX Runtime GPU." }

& $venvPython -c "import onnxruntime as ort; providers=ort.get_available_providers(); print('ONNX Runtime:', ort.__version__); print('Providers:', providers); assert 'CUDAExecutionProvider' in providers, 'CUDAExecutionProvider is unavailable'"
if ($LASTEXITCODE -ne 0) {
    throw "ONNX Runtime installed, but CUDAExecutionProvider could not be loaded."
}

Write-Host "RapidOCR GPU setup complete. Select 'RapidOCR GPU (CUDA)' in the app."
