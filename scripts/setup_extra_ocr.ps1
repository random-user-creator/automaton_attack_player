$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The project environment does not exist. Run setup.ps1 first."
}

# Install the official CUDA 12.6 PyTorch wheels used by EasyOCR.
& $venvPython -m pip install torch==2.12.1 torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch." }

& $venvPython -m pip install "easyocr>=1.7,<2" "pytesseract>=0.3.13,<0.4"
if ($LASTEXITCODE -ne 0) { throw "Failed to install EasyOCR or pytesseract." }

& $venvPython -c "import easyocr, pytesseract, torch; print('EasyOCR:', easyocr.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "EasyOCR GPU verification failed." }

$tesseractCommand = Get-Command tesseract -ErrorAction SilentlyContinue
$standardTesseract = Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"
if ($tesseractCommand) {
    Write-Host "Tesseract found at $($tesseractCommand.Source)"
} elseif (Test-Path -LiteralPath $standardTesseract) {
    Write-Host "Tesseract found at $standardTesseract"
} else {
    Write-Warning "Tesseract executable not found. Install Tesseract 5 for Windows before selecting Tesseract CPU."
}

Write-Host "Extra OCR setup complete. Restart the app and select an EasyOCR or Tesseract backend."
