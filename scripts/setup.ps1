$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

function Invoke-Python313 {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.13 @Arguments
        return
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python 3.13 was not found. Install 64-bit Python 3.13 with Tcl/Tk support."
    }
    & $python.Source @Arguments
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python 3.13 virtual environment at $venvPath"
    Invoke-Python313 -Arguments @(
        "-c",
        "import sys; assert sys.version_info[:2] == (3, 13), f'Python 3.13 required, found {sys.version}'"
    )
    if ($LASTEXITCODE -ne 0) { throw "Python 3.13 verification failed." }

    Invoke-Python313 -Arguments @("-m", "venv", $venvPath)
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

& $venvPython -c "import sys, tkinter; assert sys.version_info[:2] == (3, 13); print('Python:', sys.version); print('Tk:', tkinter.TkVersion)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.13 or Tcl/Tk support is unavailable. Repair the Python installation and recreate .venv."
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install project requirements." }

& $venvPython -m pip install --editable $projectRoot --no-deps
if ($LASTEXITCODE -ne 0) { throw "Failed to install the application package." }

& $venvPython -c "import PIL, cv2, dxcam, easyocr, mss, numpy, onnxruntime, rapidocr; print('Base dependencies imported successfully.'); print('ONNX providers:', onnxruntime.get_available_providers())"
if ($LASTEXITCODE -ne 0) { throw "Base dependency verification failed." }

Write-Host ""
Write-Host "Setup complete. EasyOCR CPU and RapidOCR CPU are ready."
Write-Host "Start the app with: .\.venv\Scripts\python.exe app.py"
Write-Host "Optional GPU backends are documented in README.md."
