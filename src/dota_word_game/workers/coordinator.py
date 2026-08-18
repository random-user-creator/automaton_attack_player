from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

from PIL import Image

from ..logging import timing_log
from ..ocr.process import ocr_process_main
from ..ocr.results import extract_words
from ..queueing import put_latest
from ..typing.process import typing_process_main
from ..vision.processing import GreenFilter

class PipelineCoordinator:
    """Own independent selectable OCR and keyboard subprocesses."""

    def __init__(self) -> None:
        self.context = mp.get_context("spawn")
        self.ocr_process: mp.Process | None = None
        self.typing_process: mp.Process | None = None
        self.active_backend: str | None = None
        self._create_ipc()

    def _create_ipc(self) -> None:
        self.stop_event = self.context.Event()
        self.frames = self.context.Queue(maxsize=1)
        self.typing_commands = self.context.Queue(maxsize=4)
        self.updates = self.context.Queue(maxsize=4)
        self.messages = self.context.Queue(maxsize=4)
        self.errors = self.context.Queue(maxsize=2)

    def start(self, backend: str = "easyocr") -> None:
        if (
            self.active_backend == backend
            and
            self.ocr_process
            and self.ocr_process.is_alive()
            and self.typing_process
            and self.typing_process.is_alive()
        ):
            return
        self.stop()
        self._create_ipc()
        self.typing_process = self.context.Process(
            target=typing_process_main,
            args=(self.stop_event, self.typing_commands, self.messages, self.errors),
            daemon=True,
            name="dota-word-typing",
        )
        self.ocr_process = self.context.Process(
            target=ocr_process_main,
            args=(
                backend,
                self.stop_event,
                self.frames,
                self.typing_commands,
                self.updates,
                self.messages,
                self.errors,
            ),
            daemon=True,
            name=f"dota-word-{backend}",
        )
        self.typing_process.start()
        self.ocr_process.start()
        self.active_backend = backend
        timing_log(
            "MAIN",
            "workers_started",
            backend=backend,
            ocr_pid=self.ocr_process.pid,
            typing_pid=self.typing_process.pid,
        )

    def stop(self) -> None:
        timing_log("MAIN", "ocr_workers_stop_start")
        self.stop_event.set()
        for process in (self.ocr_process, self.typing_process):
            if process and process.is_alive():
                timing_log("MAIN", "worker_join_start", process=process.name)
                process.join(timeout=0.5)
            if process and process.is_alive():
                timing_log("MAIN", "worker_terminate", process=process.name)
                process.terminate()
                process.join(timeout=0.5)
            if process and process.is_alive():
                timing_log("MAIN", "worker_kill", process=process.name)
                process.kill()
                process.join(timeout=0.5)
        self.ocr_process = None
        self.typing_process = None
        self.active_backend = None
        for ipc_queue in (
            self.frames,
            self.typing_commands,
            self.updates,
            self.messages,
            self.errors,
        ):
            try:
                ipc_queue.cancel_join_thread()
                ipc_queue.close()
            except (OSError, ValueError):
                pass
        timing_log("MAIN", "ocr_workers_stop_complete")

    def submit(
        self,
        detection_frame: Image.Image,
        source_frame: Any,
        filter_settings: GreenFilter,
        frame_id: int,
        captured_at: float,
        confidence: float,
        auto_type: bool,
        target_window: int | None,
        keystroke_delay_ms: float,
        recognition_scale_percent: int = 200,
        recognition_resize_method: str = "nearest",
        skip_ocr_detector: bool = True,
        crop_padding_percent: float = 5.0,
    ) -> None:
        if not self.ocr_process or not self.ocr_process.is_alive():
            return
        serialize_started = time.perf_counter()
        detection_pixels = detection_frame.tobytes()
        if isinstance(source_frame, Image.Image):
            source_mode = source_frame.mode
            source_size = source_frame.size
            source_pixels = source_frame.tobytes()
        else:
            import numpy as np

            source_array = np.asarray(source_frame, dtype=np.uint8)
            if source_array.ndim != 3 or source_array.shape[2] < 3:
                raise ValueError("OCR source must be an RGB image array.")
            source_array = np.ascontiguousarray(source_array[:, :, :3])
            source_mode = "RGB"
            source_size = (source_array.shape[1], source_array.shape[0])
            source_pixels = source_array.tobytes()
        serialized_at = time.perf_counter()
        payload = (
            frame_id,
            captured_at,
            serialized_at,
            detection_frame.mode,
            detection_frame.size,
            detection_pixels,
            source_mode,
            source_size,
            source_pixels,
            filter_settings,
            confidence,
            auto_type,
            target_window,
            keystroke_delay_ms,
            max(100, min(400, int(recognition_scale_percent))),
            (
                str(recognition_resize_method).lower()
                if str(recognition_resize_method).lower()
                in {"nearest", "box", "bilinear", "bicubic", "lanczos"}
                else "nearest"
            ),
            bool(skip_ocr_detector),
            max(0.0, min(100.0, float(crop_padding_percent))),
        )
        queue_result = put_latest(self.frames, payload)
        finished_at = time.perf_counter()
        timing_log(
            "IPC",
            "frame_submitted",
            frame=frame_id,
            serialize_ms=f"{(serialized_at - serialize_started) * 1000:.1f}",
            queue_put_ms=f"{(finished_at - serialized_at) * 1000:.1f}",
            queue_result=queue_result,
            detection_bytes=len(detection_pixels),
            source_bytes=len(source_pixels),
            bytes=len(detection_pixels) + len(source_pixels),
        )

    @staticmethod
    def extract_words(output: Any, minimum_confidence: float) -> list[str]:
        return extract_words(output, minimum_confidence)
