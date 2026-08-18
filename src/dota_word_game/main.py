from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path

from .platform.windows import enable_dpi_awareness


def _packaged_self_test() -> None:
    """Load the embedded OCR models for release-build verification."""
    from paddleocr import TextDetection, TextRecognition

    from .paths import bundled_model_dir

    detector_name = "PP-OCRv4_mobile_det"
    recognizer_name = "PP-OCRv4_mobile_rec"
    detector_dir = bundled_model_dir(detector_name)
    recognizer_dir = bundled_model_dir(recognizer_name)
    if detector_dir is None or recognizer_dir is None:
        raise RuntimeError("Embedded PaddleOCR models are missing.")
    TextDetection(model_name=detector_name, model_dir=str(detector_dir), device="cpu")
    TextRecognition(
        model_name=recognizer_name,
        model_dir=str(recognizer_dir),
        device="cpu",
    )


def main() -> None:
    # Frozen Windows worker processes execute this entry point again. Divert
    # them before importing Tkinter and the rest of the application graph.
    mp.freeze_support()
    if "--self-test" in sys.argv:
        try:
            _packaged_self_test()
        except Exception:
            diagnostic_path = Path(tempfile.gettempdir()) / (
                "AutomatonAttackPlayer-self-test.log"
            )
            diagnostic_path.write_text(
                traceback.format_exc()
                + f"\nExecutable: {sys.executable}\n"
                + f"Resource root: {getattr(sys, '_MEIPASS', None)}\n"
                + f"PATH: {os.environ.get('PATH', '')}\n",
                encoding="utf-8",
            )
            raise
        return
    from .ui.application import DotaWordGameApp

    enable_dpi_awareness()
    app = DotaWordGameApp()
    signal.signal(signal.SIGINT, lambda _signal, _frame: app.after(0, app._on_close))
    try:
        app.mainloop()
    finally:
        if not app._closing:
            app._on_close()
