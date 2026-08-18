$ErrorActionPreference = "Stop"

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildVenv = Join-Path $projectRoot ".packaging-venv"
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$releaseRequirements = Join-Path $projectRoot "packaging\requirements-release.txt"
$specFile = Join-Path $projectRoot "packaging\automaton_attack_player.spec"

function Invoke-Python313 {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.13 @Arguments
        return
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python 3.13 was not found."
    }
    & $python.Source @Arguments
}

if (-not (Test-Path -LiteralPath $buildPython)) {
    Write-Host "Creating isolated release environment at $buildVenv"
    Invoke-Python313 -Arguments @("-m", "venv", $buildVenv)
    if ($LASTEXITCODE -ne 0) { throw "Failed to create release environment." }
}

& $buildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

# Install the CPU-only PyTorch dependency before PaddleOCR. This prevents pip
# from selecting the multi-gigabyte CUDA wheel for the portable CPU release.
& $buildPython -m pip install torch==2.12.1 torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch." }

& $buildPython -m pip install -r $releaseRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install release dependencies." }

& $buildPython -m pip install --editable $projectRoot --no-deps
if ($LASTEXITCODE -ne 0) { throw "Failed to install the application package." }

$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
& $buildPython -c "from paddleocr import TextDetection, TextRecognition; TextDetection(model_name='PP-OCRv4_mobile_det', device='cpu'); TextRecognition(model_name='PP-OCRv4_mobile_rec', device='cpu'); print('Release models cached.')"
if ($LASTEXITCODE -ne 0) { throw "Failed to download or validate release models." }

$modelRoot = Join-Path $env:USERPROFILE ".paddlex\official_models"
if (-not (Test-Path -LiteralPath $modelRoot)) {
    throw "PaddleOCR model cache was not found at $modelRoot"
}

$env:AUTOMATON_PROJECT_ROOT = $projectRoot
$env:AUTOMATON_MODEL_ROOT = $modelRoot
Push-Location $projectRoot
try {
    & $buildPython -m PyInstaller --noconfirm --clean $specFile
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
}
finally {
    Pop-Location
}

$exePath = Join-Path $projectRoot "dist\AutomatonAttackPlayer.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Expected executable was not created: $exePath"
}
$sizeMb = [math]::Round((Get-Item -LiteralPath $exePath).Length / 1MB, 1)
Write-Host "Release executable: $exePath ($sizeMb MB)"

$process = Start-Process -FilePath $exePath -ArgumentList "--self-test" `
    -PassThru -Wait -WindowStyle Hidden
if ($process.ExitCode -ne 0) {
    $diagnostic = Join-Path $env:TEMP "AutomatonAttackPlayer-self-test.log"
    throw "Packaged OCR self-test failed. Diagnostic: $diagnostic"
}
Write-Host "Packaged OCR self-test passed."
