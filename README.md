# Dota Word Game

Dota Word Game is a low-latency Windows screen-capture, OCR, and auto-typing
tool. The user selects a region of a game window, the app isolates the green
word pixels, groups likely word regions, recognizes the original full-quality
crops, and types the detected letters into the selected game window.

The application is intentionally tunable. Detection can run on a very small
frame for speed while recognition still uses an unfiltered crop from the
original capture. RapidOCR, PaddleOCR, EasyOCR, and Tesseract can be compared
from the same UI.

## Platform and requirements

The application currently targets **64-bit Windows 10 or Windows 11**. DXcam,
Windows `SendInput`, foreground-window checks, and the setup scripts are
Windows-specific.

Required:

- 64-bit Python 3.13
- Tcl/Tk support from the Python installer
- PowerShell
- A game or other window running in windowed or borderless-windowed mode
- Enough memory for the selected OCR backend and full-resolution frame queue

Optional:

- An NVIDIA GPU for RapidOCR GPU, PaddleOCR GPU, or EasyOCR GPU
- A compatible NVIDIA driver and CUDA/cuDNN runtime for the selected package
- The native Tesseract 5 executable for the Tesseract backend

The Python dependencies are listed in `requirements.txt`. Important packages
include Pillow, NumPy, OpenCV, DXcam, MSS, RapidOCR, ONNX Runtime, PaddleOCR,
EasyOCR, and pytesseract.

## Quick start: PaddleOCR CPU

The default tuning profile uses PaddleOCR CPU and does not require CUDA. The
base setup also installs EasyOCR CPU and RapidOCR CPU for comparison; the
second setup command below installs PaddlePaddle's CPU runtime.

Open PowerShell in the project directory:

```powershell
cd <path-to-project>\dota_word_game
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_paddleocr_cpu.ps1
.\.venv\Scripts\python.exe app.py
```

`scripts\setup.ps1` performs the following steps:

1. Finds Python 3.13 through the Windows `py` launcher or `python` command.
2. Creates a project-local `.venv` if it does not already exist.
3. Verifies Python 3.13 and Tkinter.
4. Upgrades pip and installs `requirements.txt`.
5. Installs the `src` package in editable mode.
6. Verifies the base capture, image-processing, EasyOCR, and RapidOCR imports.

The virtual environment and local `app_state.json` are excluded by `.gitignore`.

### Manual setup

If scripts are disabled, the equivalent basic commands are:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Check Tkinter separately with:

```powershell
.\.venv\Scripts\python.exe -m tkinter
```

A small Tk window should open. If Python reports that `init.tcl` is missing,
modify or reinstall Python 3.13 and include **Tcl/Tk support and IDLE**, delete
`.venv`, and run `scripts\setup.ps1` again.

## Optional OCR backend setup

Always run `scripts\setup.ps1` first. The following scripts modify the same
`.venv`.

### RapidOCR GPU

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_rapidocr_gpu.ps1
```

This replaces CPU ONNX Runtime with the CUDA build and verifies that
`CUDAExecutionProvider` is available. The app verifies that both RapidOCR
detection and recognition use CUDA. The configured package range targets CUDA
12.x and cuDNN 9.x.

To return to ONNX Runtime CPU:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime-gpu
.\.venv\Scripts\python.exe -m pip install "onnxruntime>=1.21,<1.27"
```

### PaddleOCR CPU

PaddleOCR requires a separate PaddlePaddle runtime. To use the CPU backend:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_paddleocr_cpu.ps1
```

The CPU and GPU PaddlePaddle packages cannot coexist. This script replaces any
GPU build with PaddlePaddle CPU 3.2.2. It is the simplest PaddleOCR setup and is
often faster than GPU for the small word crops used by this app. PaddleX also
imports ModelScope and therefore PyTorch; if the existing PyTorch runtime cannot
load, the script repairs it with the official CPU build.

### PaddleOCR GPU

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_paddleocr.ps1
```

This installs the PaddlePaddle GPU 3.2.2 CUDA 12.6 build and runs Paddle's GPU
verification. If a later package change replaces it with an incompatible
build, run:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\repair_paddle_gpu.ps1
```

Both Paddle setup scripts enable two model sizes in the OCR backend menu:

- **PaddleOCR CPU/GPU** uses the small PP-OCRv4 mobile detector and recognizer.
- **PaddleOCR Server CPU/GPU** uses the larger PP-OCRv5 server detector and
  recognizer for higher accuracy. The first selection downloads about 165 MB
  of model files and takes longer to initialize. Server CPU is expected to be
  substantially slower; server GPU is the practical choice for live use.

### EasyOCR GPU

The normal requirements install EasyOCR and a CPU-capable PyTorch stack. To
install the official CUDA 12.6 PyTorch build used by this project:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup_extra_ocr.ps1
```

The script verifies `torch.cuda.is_available()` and prints the selected GPU.
EasyOCR downloads its English model files the first time it is loaded.

### Tesseract CPU

`pytesseract` is only a Python wrapper. Install the native Tesseract 5 Windows
executable by following the
[official Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).

The app searches the system `PATH` and the standard location:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Selecting an area and controlling capture

1. Start the app with `.\.venv\Scripts\python.exe app.py`.
2. Open the game and place it where it will remain while capturing.
3. Click **select_area**. The Tkinter window temporarily hides.
4. Drag a rectangle around the part of the game where target words can appear.
   Press `Esc` to cancel.
5. Capture and OCR start automatically after a successful selection.

The window under the center of the selected rectangle becomes the typing
target. Auto-typing is allowed only while that same window is the active
foreground window. This prevents recognized text from being typed into another
program after Alt+Tab.

Select as tightly as practical. A smaller area reduces GPU/CPU capture work,
native-frame serialization, filtering, and false green scenery candidates.
Multiple words may still be captured and recognized at the same time.

Capture controls:

- **Stop** stops new screen frames but keeps the selected rectangle and loaded
  OCR model.
- **Start** resumes the saved rectangle without opening the selector again.
- **select_area** replaces the previous rectangle with a new one.
- Closing the window or pressing `Ctrl+C` saves the current settings cleanly.

The next launch restores the capture rectangle and settings but waits for
**Start**. If the previously saved game-window handle is no longer valid, the
app resolves the window under the center of the saved rectangle again.

## How the pipeline works

The latency-sensitive work is separated into independent execution sections:

- Tkinter UI in the main process
- Buffered DXcam capture and NumPy/OpenCV filtering in a producer thread
- Detection and recognition in a spawned OCR process
- Windows `SendInput` keyboard output in a separate process

For every submitted frame:

1. DXcam supplies the newest RGB NumPy frame. MSS and Pillow are fallbacks.
2. The frame is reduced to the configured **Processing resolution**.
3. HSV thresholds create a one-channel green mask.
4. Optional erosion, dilation, and text-row cleanup modify the mask.
5. Nearby white components become grouped blob candidates.
6. The OCR detector either examines the compact candidate canvas, or the app
   skips detection and treats each blob group as a word box.
7. Candidate coordinates are mapped to the original unfiltered RGB frame.
8. The mapped crop is padded, optionally upscaled, and passed to recognition.
9. Accepted ASCII-letter words are sent to the independent typing process.

Queues favor the newest frame. When processing cannot keep up, stale frames are
replaced rather than allowed to build a growing latency backlog. Frames with no
blob groups skip detection and recognition entirely.

## Tuning reference

Defaults shown below apply to a fresh configuration. Once the app has run, the
saved values in `app_state.json` take precedence.

### Capture and resolution

| Setting | Range / default | Explanation |
|---|---|---|
| FPS | 1–60; default 15 | How often the app attempts to process a frame. Higher values react sooner only if capture and OCR can keep up. |
| Processing resolution | 10–100%; default 15% | Reduces the frame before green filtering, blob grouping, and OCR detection. Lower values make blob detection much faster, but very small letters such as `IO` may disappear or merge. Recognition still uses the original frame. |
| Resize method | Nearest by default | Controls how the low-resolution detection image is produced. It does not resize the native recognition crop. |
| REC crop scale | 100–400%; default 250% | Upscales the original unfiltered crop immediately before recognition. This may help a recognizer with small words, but it cannot create missing detail and increases recognition time. |
| REC resize method | Bicubic by default | Interpolation used only when REC crop scale is above 100%. It is independent from the detection resize method. |
| REC crop padding | 0–100%; default 5% | Expands the mapped crop independently on left, right, top, and bottom before resizing and recognition. Five percent per side produces a crop roughly 10% wider and taller. |

Detection resize methods:

| Method | Tradeoff |
|---|---|
| Nearest | Fastest NumPy path and preserves exact colors, but thin strokes can alias or disappear at very low resolution. |
| Box | Good general downscaler; averages source pixels and is still fast. |
| Bilinear | Smooth and moderate speed, but may soften very thin mask strokes. |
| Lanczos | Highest-cost detection resize; sharper detail but can introduce ringing and usually is unnecessary for a binary mask. |

Recognition resize methods:

| Method | Tradeoff |
|---|---|
| Nearest | Fast and blocky. Useful for pixel-like text or experimentation. |
| Box | Primarily intended for downscaling; usually not the best REC upscaler. |
| Bilinear | Fast, smooth upscaling. |
| Bicubic | Recommended balance for anti-aliased game text. |
| Lanczos | Sharp but slower; may create halos around letter edges. |

A good first test for small text is `150–200%` REC scale, Bicubic, and 5%
padding. Compare accuracy and `recognition_ms` against 100% before increasing
further.

### Green mask and blob grouping

| Setting | Range / default | Explanation |
|---|---|---|
| Show only green text | On | Enables HSV masking. Matching pixels become white and everything else becomes black in the preview. Turning it off also disables blob-based detector skipping. |
| Clean text rows | On | Retains horizontal, letter-like bands while rejecting sparse or large solid scenery regions. Supports multiple rows when enabled. |
| Hue range | 80–93° | Target green hue interval sampled from the reference game screenshot. |
| Minimum saturation | 35% | Rejects gray or weakly colored pixels. Increase it when pale scenery is included; decrease it if letter edges vanish. |
| Minimum brightness | 50% | Rejects dark green pixels. Increase it for dark background noise; decrease it if dim letters disappear. |
| Erode | 0–5; default 0 | A 3×3 erosion removes white edge pixels and isolated noise. Even one iteration can erase thin letters at low processing resolution. |
| Dilate | 0–5; default 1 | A 3×3 dilation thickens and joins nearby white pixels. One iteration can repair broken strokes, but too much can merge letters or words. |
| Minimum grouped blob area | 0–100000; default 10 | Rejects groups with too few white pixels at the processed resolution. Lower resolutions produce much smaller area values. Keep this low when two-letter words such as `IO` must survive. Zero disables the area threshold. |

Color alone cannot distinguish letters from scenery with the same hue,
saturation, and brightness. Prefer a tighter selected area and row cleanup
before aggressively increasing erosion or minimum blob area.

### OCR and direct blob recognition

| Setting | Default | Explanation |
|---|---|---|
| OCR backend | PaddleOCR CPU | Selects the detector and recognizer implementation. Switching backend restarts the OCR process and loads the selected models. |
| Minimum confidence | 0.50 | Only recognition results at or above this confidence are accepted. Lower values catch harder text but increase incorrect typing. |
| Recognize blob groups directly | On | Skips the OCR detector and sends each grouped blob's original-frame crop directly to recognition. This removes `detection_ms`, but mask noise also reaches recognition. Requires the green filter. |
| Auto-type detected words | On | Sends accepted words to the typing process. Turn it off while tuning OCR visually. |
| Keystroke delay | 0.1 ms | Delay between letters. At 0 ms, all scan-code events are submitted in one Windows `SendInput` batch. |

The direct-blob option is often fastest when blob grouping already outlines one
word per red rectangle. Leave it off when blob groups contain several unrelated
objects or when the OCR detector is needed to refine word boundaries.

### OCR backend comparison

| Backend | Setup | Advantages | Tradeoffs |
|---|---|---|---|
| RapidOCR CPU | `scripts/setup.ps1` | Fast alternative; mobile PP-OCRv4 models, predictable CPU latency, no CUDA transfer overhead. | CPU detection and recognition still cost time; accuracy depends on stylized text and crop quality. |
| RapidOCR GPU | `scripts/setup_rapidocr_gpu.ps1` | ONNX Runtime CUDA; can help larger inputs or heavier workloads. | CUDA setup is stricter, and tiny dynamic crops may be slower than CPU because launch and transfer overhead dominate. |
| PaddleOCR CPU | `scripts/setup_paddleocr_cpu.ps1` | Native Paddle PP-OCRv4 mobile models without CUDA transfer overhead; straightforward CPU comparison. | Installs the CPU PaddlePaddle runtime and therefore replaces PaddlePaddle GPU in the same environment. |
| PaddleOCR GPU | `scripts/setup_paddleocr.ps1` | Native Paddle PP-OCRv4 mobile detector/recognizer and good general OCR accuracy. | Heaviest setup; CUDA/cuDNN compatibility matters; model startup and GPU inference may have latency spikes. |
| PaddleOCR Server CPU | `scripts/setup_paddleocr_cpu.ps1` | Larger PP-OCRv5 server models prioritize detection and recognition accuracy. | Roughly 165 MB of models and much slower CPU inference; mainly useful as an accuracy comparison. |
| PaddleOCR Server GPU | `scripts/setup_paddleocr.ps1` | Higher-accuracy PP-OCRv5 server models with GPU acceleration. | Larger download, slower startup, more VRAM, and higher per-frame latency than mobile models. |
| EasyOCR CPU | `scripts/setup.ps1` | Works well with direct recognition of the original, upscaled blob crops. | Larger runtime and usually slower than mobile RapidOCR on CPU. |
| EasyOCR GPU | `scripts/setup_extra_ocr.ps1` | CUDA recognition can help larger or multiple crops. | Model loading, transfers, and shape-dependent GPU work can make small live crops slower or less consistent than CPU. |
| Tesseract CPU | `scripts/setup.ps1` plus native Tesseract 5 | Mature external engine; no neural GPU runtime. | Separate executable required; often weaker on stylized, animated game text and repeated real-time calls. |

GPU utilization does not need to reach 100% for this workload. The app submits
small, irregular images one frame at a time, so CPU preparation, GPU launch
overhead, memory transfer, and synchronization can dominate. Always compare
measured end-to-end latency rather than Task Manager utilization alone.

## Auto-typing behavior and safety

Before sending anything, recognized text is reduced to ASCII letters `A–Z` and
`a–z`. Spaces, punctuation, commas, digits, and Enter are never emitted. Words
are typed consecutively without spaces. A short cooldown prevents the same
visible word from being typed on every frame.

Keyboard output uses native Windows scan-code `SendInput`. The typing process
checks that the selected game window is focused before each command. Test with
**Auto-type detected words** disabled until the filter, confidence, and crop
settings are reliable.

Use automation only where it is permitted by the target application or game.

## Saved state

The app writes `app_state.json` beside `app.py` during normal close and
`Ctrl+C`. It stores:

- Window size, position, and maximized state
- Capture rectangle and target window handle
- FPS, processing resolution, and detection resize method
- REC scale, resize method, and crop padding
- OCR backend, confidence, detector-skip option, and auto-type setting
- Keystroke delay
- HSV, row cleanup, morphology, and minimum blob settings

The file is local runtime state and is ignored by Git. Delete it while the app
is closed to restore fresh defaults.

## Latency logs

Verbose frame-correlated timing is enabled by default. Important events include:

- `[CAPTURE]`: `grab_ms`, `convert_ms`, `scale_ms`, `filter_ms`, `submit_ms`, and `total_ms`
- `[IPC]`: serialized detection/source byte counts and queue replacement
- `[OCR]`: queue, decode, compaction, detection, crop, recognition, and end-to-end time
- `[TYPE]`: typing queue, typing duration, character count, and capture-to-type time
- `[UI]`: preview and OCR-display delay

When direct blob recognition is enabled, successful frames show:

```text
detection_ms=0.0 detector_skipped=True
```

Frames without candidates show:

```text
blob_groups=0 skip_reason=no_blobs
```

Disable verbose logging for normal use:

```powershell
$env:DOTA_WORD_VERBOSE = "0"
.\.venv\Scripts\python.exe app.py
```

Remove the variable or set it to `1` to restore logs.

## Troubleshooting

### No capture or the wrong monitor is captured

DXcam is used for regions fully inside the primary monitor. Other regions fall
back to MSS and then Pillow. Confirm the `[CAPTURE] event=backend_ready` log and
select the area again. Borderless-windowed mode is generally the easiest capture
mode to troubleshoot.

### Green letters disappear

- Use Erode 0.
- Raise Processing resolution.
- Try Box or Bilinear instead of Nearest if downsampling skips thin strokes.
- Lower minimum saturation or brightness slightly.
- Keep Minimum grouped blob area low for small words such as `IO`.
- Try Dilate 1 when strokes are fragmented.

### Green scenery becomes a blob

- Select a tighter area.
- Enable Clean text rows.
- Narrow the hue interval or raise saturation/brightness carefully.
- Increase minimum blob area only if required small words still survive.
- Leave direct blob recognition off so the OCR detector can reject the noise.

### OCR detects a blob but returns no word

- Inspect the red rectangle in the preview.
- Increase REC crop padding to avoid clipped strokes.
- Test 150–200% REC crop scale with Bicubic.
- Lower confidence temporarily while diagnosing.
- Compare RapidOCR CPU and EasyOCR CPU.
- If direct blob recognition is enabled, turn it off and compare the detector path.

### GPU is slower than CPU

This is expected for some small crops. Use the CPU backend if its measured
`capture_to_ocr_ms` is lower. GPU backends become more attractive as crop size,
crop count, or model workload increases.

### First backend load is slow

OCR packages download and initialize model files on their first use. Later
launches reuse the cache. Model loading occurs when the backend is first started
or changed, not on every frame.

### Tesseract executable was not found

Install native Tesseract 5 and restart the app. Installing `pytesseract` alone
is not sufficient.

## Project files

```text
app.py                              Small development launcher
pyproject.toml                      Package metadata and console entry point
requirements.txt                    Runtime dependencies
scripts/                            Setup and optional backend utilities
  setup.ps1                         Base Python 3.13 environment setup
  setup_rapidocr_gpu.ps1            Optional ONNX Runtime CUDA setup
  setup_paddleocr_cpu.ps1           Optional PaddleOCR CPU setup
  setup_paddleocr.ps1               Optional PaddleOCR CUDA setup
  setup_extra_ocr.ps1               Optional EasyOCR CUDA setup
  repair_paddle_gpu.ps1             Paddle GPU repair utility
src/dota_word_game/                 Importable application package
  main.py                           Application lifecycle entry point
  paths.py                          Project and saved-state locations
  logging.py                        Structured latency logging
  queueing.py                       Latest-value IPC queue helper
  capture/                          Region model, selector, and capture worker
  vision/                           Resize, HSV mask, and blob preprocessing
  ocr/                              OCR results, crop mapping, and OCR process
  typing/                           Windows SendInput typing process
  workers/                          OCR/typing process coordinator
  ui/                               Tkinter application, layout, and UI constants
app_state.json                      Generated local state; ignored by Git
```

The root launcher keeps the original command working. After `scripts/setup.ps1`
installs the package in editable mode, either of these starts the same app:

```powershell
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m dota_word_game
```
