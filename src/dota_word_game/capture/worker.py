from __future__ import annotations

import ctypes
import queue
import threading
import time

from PIL import Image, ImageGrab, ImageOps

from ..logging import timing_log
from ..vision.processing import isolate_green_text_array, resize_rgb_array
from .models import CaptureRegion

class CaptureWorker:
    def __init__(self) -> None:
        self.frames: queue.Queue[tuple[int, float, Image.Image]] = queue.Queue(maxsize=1)
        self.errors: queue.Queue[str] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(
        self,
        region: CaptureRegion,
        fps_provider,
        scale_provider,
        resize_method_provider,
        filter_provider,
        ocr_submitter,
    ) -> None:
        self.stop()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._capture_loop,
            args=(
                region,
                fps_provider,
                scale_provider,
                resize_method_provider,
                filter_provider,
                ocr_submitter,
            ),
            daemon=True,
            name="screen-capture",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            timing_log("MAIN", "capture_thread_join_start")
            self.thread.join(timeout=1.0)
            timing_log(
                "MAIN",
                "capture_thread_join_complete",
                still_alive=self.thread.is_alive(),
            )
        self.thread = None

    def _capture_loop(
        self,
        region: CaptureRegion,
        fps_provider,
        scale_provider,
        resize_method_provider,
        filter_provider,
        ocr_submitter,
    ) -> None:
        capture_backend = FastScreenCapture(region)
        timing_log("CAPTURE", "backend_ready", backend=capture_backend.name)
        frame_id = 0
        try:
            while not self.stop_event.is_set():
                started = time.perf_counter()
                frame_id += 1
                try:
                    source_array, native_grab_ms, convert_ms = capture_backend.grab()
                    captured_at = time.perf_counter()
                    scale_started = captured_at
                    processed = resize_rgb_array(
                        source_array,
                        scale_provider(),
                        resize_method_provider(),
                    )
                    scaled_at = time.perf_counter()
                    filter_settings = filter_provider()
                    filtered = isolate_green_text_array(processed, filter_settings)
                    frame = Image.fromarray(
                        filtered,
                        "L" if filter_settings.enabled else "RGB",
                    )
                    filtered_at = time.perf_counter()
                    ocr_frame = (
                        ImageOps.invert(frame) if filter_settings.enabled else frame.copy()
                    )
                    ocr_submitter(
                        ocr_frame,
                        source_array,
                        filter_settings,
                        frame_id,
                        captured_at,
                    )
                    submitted_at = time.perf_counter()
                    preview_replaced = False
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                            preview_replaced = True
                        except queue.Empty:
                            pass
                    self.frames.put_nowait((frame_id, captured_at, frame))
                    completed_at = time.perf_counter()
                    fps = max(1, min(60, fps_provider()))
                    timing_log(
                        "CAPTURE",
                        "frame_complete",
                        frame=frame_id,
                        backend=capture_backend.name,
                        grab_ms=f"{native_grab_ms:.1f}",
                        convert_ms=f"{convert_ms:.1f}",
                        scale_ms=f"{(scaled_at - scale_started) * 1000:.1f}",
                        filter_ms=f"{(filtered_at - scaled_at) * 1000:.1f}",
                        submit_ms=f"{(submitted_at - filtered_at) * 1000:.1f}",
                        total_ms=f"{(completed_at - started) * 1000:.1f}",
                        size=f"{frame.width}x{frame.height}",
                        resize_method=resize_method_provider(),
                        budget_ms=f"{1000 / fps:.1f}",
                        preview_replaced=preview_replaced,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    try:
                        self.errors.put_nowait(str(exc))
                    except queue.Full:
                        pass
                    timing_log(
                        "CAPTURE",
                        "capture_failed",
                        backend=capture_backend.name,
                        error=repr(exc),
                    )
                    self.stop_event.set()
                    return

                fps = max(1, min(60, fps_provider()))
                remaining = (1.0 / fps) - (time.perf_counter() - started)
                self.stop_event.wait(max(0.0, remaining))
        finally:
            capture_backend.close()


class FastScreenCapture:
    """Capture a region using DXGI, with robust lower-level fallbacks."""

    def __init__(self, region: CaptureRegion) -> None:
        self.region = region
        self.name = "pillow"
        self.camera = None
        self.mss_instance = None
        self.dxcam_region: tuple[int, int, int, int] | None = None
        self._initialize()

    def _initialize(self) -> None:
        # DXcam's default camera targets the primary output. Virtual desktop
        # coordinates on the primary display begin at (0, 0).
        if hasattr(ctypes, "windll"):
            primary_width = ctypes.windll.user32.GetSystemMetrics(0)
            primary_height = ctypes.windll.user32.GetSystemMetrics(1)
            inside_primary = (
                self.region.left >= 0
                and self.region.top >= 0
                and self.region.left + self.region.width <= primary_width
                and self.region.top + self.region.height <= primary_height
            )
            if inside_primary:
                try:
                    import dxcam

                    self.camera = dxcam.create(
                        region=self.region.bbox,
                        output_color="RGB",
                        processor_backend="numpy",
                    )
                    self.camera.start(
                        region=self.region.bbox,
                        target_fps=60,
                        video_mode=True,
                    )
                    self.dxcam_region = self.region.bbox
                    self.name = "dxcam-buffered"
                    return
                except Exception as exc:
                    timing_log("CAPTURE", "dxcam_unavailable", error=repr(exc))
                    self.camera = None

        try:
            import mss

            self.mss_instance = mss.mss()
            self.name = "mss"
        except Exception as exc:
            timing_log("CAPTURE", "mss_unavailable", error=repr(exc))
            self.mss_instance = None
            self.name = "pillow"

    def grab(self):
        grab_started = time.perf_counter()
        if self.camera is not None:
            array = self.camera.grab(copy=True)
            if array is None:
                array = self.camera.get_latest_frame(copy=True)
            grabbed_at = time.perf_counter()
            if array is None:
                raise RuntimeError("DXcam did not return a frame.")
        elif self.mss_instance is not None:
            screenshot = self.mss_instance.grab(
                {
                    "left": self.region.left,
                    "top": self.region.top,
                    "width": self.region.width,
                    "height": self.region.height,
                }
            )
            grabbed_at = time.perf_counter()
            import numpy as np

            bgra = np.asarray(screenshot, dtype=np.uint8)
            array = np.ascontiguousarray(bgra[:, :, 2::-1])
        else:
            screenshot = ImageGrab.grab(bbox=self.region.bbox, all_screens=True)
            grabbed_at = time.perf_counter()
            import numpy as np

            array = np.asarray(screenshot.convert("RGB"))

        converted_at = time.perf_counter()
        return (
            array,
            (grabbed_at - grab_started) * 1000,
            (converted_at - grabbed_at) * 1000,
        )

    def close(self) -> None:
        if self.camera is not None:
            try:
                if self.camera.is_capturing:
                    self.camera.stop()
                self.camera.release()
            except Exception as exc:
                timing_log("CAPTURE", "dxcam_release_failed", error=repr(exc))
            self.camera = None
        if self.mss_instance is not None:
            try:
                self.mss_instance.close()
            except Exception as exc:
                timing_log("CAPTURE", "mss_close_failed", error=repr(exc))
            self.mss_instance = None
