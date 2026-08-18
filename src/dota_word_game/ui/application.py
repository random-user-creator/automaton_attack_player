from __future__ import annotations

import json
import queue
import time
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from ..capture.models import CaptureRegion
from ..capture.selector import RegionSelector
from ..capture.worker import CaptureWorker
from ..logging import timing_log
from ..paths import STATE_PATH
from ..platform.windows import (
    focus_window,
    is_window_focused,
    is_window_valid,
    window_at_point,
)
from ..vision.processing import GreenFilter, normalize_resize_method
from ..workers.coordinator import PipelineCoordinator
from .constants import (
    REC_RESIZE_LABEL_METHODS,
    REC_RESIZE_METHOD_LABELS,
    RESIZE_LABEL_METHODS,
    RESIZE_METHOD_LABELS,
)
from .layout import UILayoutMixin


class DotaWordGameApp(UILayoutMixin, tk.Tk):
    def __init__(self) -> None:
        self.state_path = STATE_PATH
        saved = self._load_state()
        super().__init__()
        self.title("Dota Word Game")
        self.last_normal_geometry = str(saved.get("window_geometry", "960x700"))
        try:
            self.geometry(self.last_normal_geometry)
        except tk.TclError:
            self.last_normal_geometry = "960x700"
            self.geometry(self.last_normal_geometry)
        self.minsize(640, 480)

        self.region = self._region_from_state(saved.get("capture_region"))
        saved_target = saved.get("target_window")
        try:
            saved_target = int(saved_target)
        except (TypeError, ValueError):
            saved_target = None
        self.target_window = saved_target if is_window_valid(saved_target) else None
        self.worker = CaptureWorker()
        self.ocr_worker = PipelineCoordinator()
        self.fps_value = max(1, min(60, int(saved.get("fps", 15))))
        self.scale_value = max(10, min(100, int(saved.get("scale_percent", 15))))
        self.recognition_scale_value = max(
            100, min(400, int(saved.get("recognition_scale_percent", 250)))
        )
        self.crop_padding_percent_value = max(
            0.0, min(100.0, float(saved.get("crop_padding_percent", 5.0)))
        )
        saved_rec_resize_method = str(
            saved.get("recognition_resize_method", "bicubic")
        ).lower()
        self.recognition_resize_method_value = (
            saved_rec_resize_method
            if saved_rec_resize_method in REC_RESIZE_METHOD_LABELS
            else "nearest"
        )
        self.resize_method_value = normalize_resize_method(
            saved.get("resize_method", "nearest")
        )
        self.confidence_value = max(
            0.0, min(1.0, float(saved.get("confidence", 0.50)))
        )
        self.auto_type_value = bool(saved.get("auto_type", True))
        self.skip_ocr_detector_value = bool(
            saved.get("skip_ocr_detector", True)
        )
        saved_backend = str(saved.get("ocr_backend", "paddle_cpu"))
        self.ocr_backend_value = (
            saved_backend
            if saved_backend
            in {
                "rapidocr",
                "rapidocr_gpu",
                "paddle_cpu",
                "paddle",
                "paddle_server_cpu",
                "paddle_server_gpu",
                "easyocr",
                "easyocr_gpu",
                "tesseract",
            }
            else "paddle_cpu"
        )
        self.keystroke_delay_ms = max(
            0.0, min(1000.0, float(saved.get("keystroke_delay_ms", 0.1)))
        )
        filter_state = saved.get("green_filter", {})
        self.green_filter = GreenFilter(
            enabled=bool(filter_state.get("enabled", True)),
            keep_text_bands=bool(filter_state.get("keep_text_bands", True)),
            hue_min=max(0, min(359, int(filter_state.get("hue_min", 80)))),
            hue_max=max(0, min(359, int(filter_state.get("hue_max", 93)))),
            saturation_min=max(
                0, min(100, int(filter_state.get("saturation_min", 35)))
            ),
            value_min=max(0, min(100, int(filter_state.get("value_min", 50)))),
            erosion_iterations=max(
                0, min(5, int(filter_state.get("erosion_iterations", 0)))
            ),
            dilation_iterations=max(
                0, min(5, int(filter_state.get("dilation_iterations", 1)))
            ),
            minimum_blob_area=max(
                0, min(100000, int(filter_state.get("minimum_blob_area", 10)))
            ),
        )
        self.fps_var = tk.IntVar(value=self.fps_value)
        self.scale_var = tk.DoubleVar(value=self.scale_value)
        self.scale_label_var = tk.StringVar(value=f"{self.scale_value}%")
        self.recognition_scale_var = tk.DoubleVar(
            value=self.recognition_scale_value
        )
        self.recognition_scale_label_var = tk.StringVar(
            value=f"{self.recognition_scale_value}%"
        )
        self.recognition_resize_method_var = tk.StringVar(
            value=REC_RESIZE_METHOD_LABELS[
                self.recognition_resize_method_value
            ]
        )
        self.crop_padding_percent_var = tk.DoubleVar(
            value=self.crop_padding_percent_value
        )
        self.resize_method_var = tk.StringVar(
            value=RESIZE_METHOD_LABELS[self.resize_method_value]
        )
        self.confidence_var = tk.DoubleVar(value=self.confidence_value)
        self.auto_type_var = tk.BooleanVar(value=self.auto_type_value)
        self.skip_ocr_detector_var = tk.BooleanVar(
            value=self.skip_ocr_detector_value
        )
        self.ocr_backend_var = tk.StringVar(
            value={
                "rapidocr": "RapidOCR CPU",
                "rapidocr_gpu": "RapidOCR GPU (CUDA)",
                "paddle_cpu": "PaddleOCR CPU",
                "paddle": "PaddleOCR GPU",
                "paddle_server_cpu": "PaddleOCR Server CPU",
                "paddle_server_gpu": "PaddleOCR Server GPU",
                "easyocr": "EasyOCR CPU",
                "easyocr_gpu": "EasyOCR GPU (CUDA)",
                "tesseract": "Tesseract CPU",
            }[self.ocr_backend_value]
        )
        self.keystroke_delay_var = tk.DoubleVar(value=self.keystroke_delay_ms)
        self.filter_enabled_var = tk.BooleanVar(value=self.green_filter.enabled)
        self.keep_text_bands_var = tk.BooleanVar(
            value=self.green_filter.keep_text_bands
        )
        self.hue_min_var = tk.IntVar(value=self.green_filter.hue_min)
        self.hue_max_var = tk.IntVar(value=self.green_filter.hue_max)
        self.saturation_min_var = tk.IntVar(value=self.green_filter.saturation_min)
        self.value_min_var = tk.IntVar(value=self.green_filter.value_min)
        self.erosion_iterations_var = tk.IntVar(
            value=self.green_filter.erosion_iterations
        )
        self.dilation_iterations_var = tk.IntVar(
            value=self.green_filter.dilation_iterations
        )
        self.minimum_blob_area_var = tk.IntVar(
            value=self.green_filter.minimum_blob_area
        )
        self.region_var = tk.StringVar(value="No area selected")
        self.status_var = tk.StringVar(value="Select an area to begin capturing.")
        self.ocr_status_var = tk.StringVar(value="OCR stopped")
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.latest_blob_boxes: tuple[tuple[int, int, int, int], ...] = ()
        self.latest_blob_detection_size: tuple[int, int] = (0, 0)
        self.saved_window_geometry: str | None = None
        self.saved_window_state = "normal"
        self._closing = False

        self._build_ui()
        if self.region:
            self.region_var.set(
                f"Area: x={self.region.left}, y={self.region.top}, "
                f"{self.region.width} × {self.region.height}px"
            )
            self.status_var.set("Saved area restored. Press Start to capture.")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._remember_window_geometry, add=True)
        if saved.get("window_state") == "zoomed":
            self.after_idle(lambda: self.state("zoomed"))
        self.after(15, self._display_latest_frame)
        self.after(50, self._poll_ocr)

    def _load_state(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _region_from_state(data) -> CaptureRegion | None:
        if not isinstance(data, dict):
            return None
        try:
            region = CaptureRegion(
                left=int(data["left"]),
                top=int(data["top"]),
                width=int(data["width"]),
                height=int(data["height"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return region if region.width > 1 and region.height > 1 else None

    def _open_region_selector(self) -> None:
        self.worker.stop()
        self.status_var.set("Drag over the portion of the screen to capture.")
        self.update_idletasks()
        self.saved_window_geometry = self.geometry()
        self.saved_window_state = self.state()
        self.withdraw()
        # Let the app window disappear before the selector takes a screenshot-sized area.
        self.after(150, self._show_selector)

    def _show_selector(self) -> None:
        selector = RegionSelector(self, self._area_selected)
        selector.bind("<Destroy>", self._selector_closed, add=True)

    def _selector_closed(self, event: tk.Event) -> None:
        if event.widget.winfo_toplevel() is event.widget:
            self._restore_main_window()
            if self.region is None:
                self.status_var.set("Selection cancelled. Select an area to begin.")

    def _restore_main_window(self) -> None:
        self.deiconify()
        self.state("normal")
        if self.saved_window_geometry:
            self.geometry(self.saved_window_geometry)
        if self.saved_window_state == "zoomed":
            self.state("zoomed")
        self.lift()

    def _area_selected(self, region: CaptureRegion, target_window: int | None) -> None:
        self.region = region
        self.target_window = target_window
        self.region_var.set(
            f"Area: x={region.left}, y={region.top}, "
            f"{region.width} × {region.height}px"
        )
        self._start_capture()

    def _start_capture(self) -> None:
        if not self.region:
            self.status_var.set("Select an area before starting capture.")
            return
        self._resolve_target_window()
        self._apply_ocr_settings()
        self.ocr_worker.start(self.ocr_backend_value)
        self.worker.start(
            self.region,
            lambda: self.fps_value,
            lambda: self.scale_value,
            lambda: self.resize_method_value,
            lambda: self.green_filter,
            lambda detection_frame, source_array, filter_settings, frame_id, captured_at: self.ocr_worker.submit(
                detection_frame,
                source_array,
                filter_settings,
                frame_id,
                captured_at,
                self.confidence_value,
                self.auto_type_value,
                self.target_window,
                self.keystroke_delay_ms,
                self.recognition_scale_value,
                self.recognition_resize_method_value,
                self.skip_ocr_detector_value,
                self.crop_padding_percent_value,
            ),
        )
        self._update_capture_status()
        self.after(150, lambda: focus_window(self.target_window))

    def _resolve_target_window(self) -> None:
        if not self.region or is_window_valid(self.target_window):
            return
        self.update_idletasks()
        self.saved_window_geometry = self.geometry()
        self.saved_window_state = self.state()
        self.withdraw()
        self.update_idletasks()
        self.target_window = window_at_point(
            self.region.left + self.region.width // 2,
            self.region.top + self.region.height // 2,
        )
        self._restore_main_window()

    def _apply_fps(self, _event=None) -> None:
        try:
            value = int(self.fps_var.get())
        except (tk.TclError, ValueError):
            value = self.fps_value
        self.fps_value = max(1, min(60, value))
        self.fps_var.set(self.fps_value)
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_scale(self, value=None) -> None:
        if value is None:
            value = self.scale_var.get()
        self.scale_value = max(10, min(100, round(float(value) / 5) * 5))
        self.scale_var.set(self.scale_value)
        self.scale_label_var.set(f"{self.scale_value}%")
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_recognition_scale(self, value=None) -> None:
        if value is None:
            value = self.recognition_scale_var.get()
        self.recognition_scale_value = max(
            100, min(400, round(float(value)))
        )
        self.recognition_scale_var.set(self.recognition_scale_value)
        self.recognition_scale_label_var.set(
            f"{self.recognition_scale_value}%"
        )
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_resize_method(self, _event=None) -> None:
        selected = self.resize_method_var.get()
        self.resize_method_value = RESIZE_LABEL_METHODS.get(selected, "nearest")
        self.resize_method_var.set(
            RESIZE_METHOD_LABELS[self.resize_method_value]
        )
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_recognition_resize_method(self, _event=None) -> None:
        selected = self.recognition_resize_method_var.get()
        self.recognition_resize_method_value = REC_RESIZE_LABEL_METHODS.get(
            selected, "bicubic"
        )
        self.recognition_resize_method_var.set(
            REC_RESIZE_METHOD_LABELS[self.recognition_resize_method_value]
        )
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_crop_padding(self, _event=None) -> None:
        try:
            value = float(self.crop_padding_percent_var.get())
        except (tk.TclError, ValueError):
            value = self.crop_padding_percent_value
        self.crop_padding_percent_value = max(0.0, min(100.0, value))
        self.crop_padding_percent_var.set(self.crop_padding_percent_value)
        if self.region and not self.worker.stop_event.is_set():
            self._update_capture_status()

    def _apply_ocr_settings(self, _event=None) -> None:
        try:
            confidence = float(self.confidence_var.get())
        except (tk.TclError, ValueError):
            confidence = self.confidence_value
        self.confidence_value = max(0.0, min(1.0, confidence))
        self.confidence_var.set(round(self.confidence_value, 2))
        self.auto_type_value = bool(self.auto_type_var.get())
        self.skip_ocr_detector_value = bool(
            self.skip_ocr_detector_var.get()
        )
        try:
            keystroke_delay = float(self.keystroke_delay_var.get())
        except (tk.TclError, ValueError):
            keystroke_delay = self.keystroke_delay_ms
        self.keystroke_delay_ms = max(0.0, min(1000.0, keystroke_delay))
        self.keystroke_delay_var.set(self.keystroke_delay_ms)

    def _change_ocr_backend(self, _event=None) -> None:
        selected = self.ocr_backend_var.get()
        backend = {
            "RapidOCR CPU": "rapidocr",
            "RapidOCR GPU (CUDA)": "rapidocr_gpu",
            "PaddleOCR CPU": "paddle_cpu",
            "PaddleOCR GPU": "paddle",
            "PaddleOCR Server CPU": "paddle_server_cpu",
            "PaddleOCR Server GPU": "paddle_server_gpu",
            "EasyOCR CPU": "easyocr",
            "EasyOCR GPU (CUDA)": "easyocr_gpu",
            "Tesseract CPU": "tesseract",
        }.get(selected, "rapidocr")
        if backend == self.ocr_backend_value:
            return
        was_capturing = bool(self.worker.thread and self.worker.thread.is_alive())
        self.worker.stop()
        self.ocr_worker.stop()
        self.ocr_backend_value = backend
        self.ocr_status_var.set(f"Switched to {selected}; loading model…")
        if was_capturing:
            self.after(10, self._start_capture)

    def _update_capture_status(self) -> None:
        if not self.region:
            return
        width = max(1, round(self.region.width * self.scale_value / 100))
        height = max(1, round(self.region.height * self.scale_value / 100))
        self.status_var.set(
            f"Capturing at {self.fps_value} FPS • "
            f"processing {width} × {height}px ({self.scale_value}%, "
            f"{self.resize_method_value}) • REC crop "
            f"{self.recognition_scale_value}% "
            f"{self.recognition_resize_method_value} • pad "
            f"{self.crop_padding_percent_value:g}%"
        )

    def _apply_filter(self, _event=None) -> None:
        try:
            hue_min = max(0, min(359, int(self.hue_min_var.get())))
            hue_max = max(0, min(359, int(self.hue_max_var.get())))
            saturation_min = max(
                0, min(100, int(self.saturation_min_var.get()))
            )
            value_min = max(0, min(100, int(self.value_min_var.get())))
            erosion_iterations = max(
                0, min(5, int(self.erosion_iterations_var.get()))
            )
            dilation_iterations = max(
                0, min(5, int(self.dilation_iterations_var.get()))
            )
            minimum_blob_area = max(
                0, min(100000, int(self.minimum_blob_area_var.get()))
            )
        except (tk.TclError, ValueError):
            return
        if hue_min > hue_max:
            hue_min, hue_max = hue_max, hue_min

        self.green_filter = GreenFilter(
            enabled=self.filter_enabled_var.get(),
            keep_text_bands=self.keep_text_bands_var.get(),
            hue_min=hue_min,
            hue_max=hue_max,
            saturation_min=saturation_min,
            value_min=value_min,
            erosion_iterations=erosion_iterations,
            dilation_iterations=dilation_iterations,
            minimum_blob_area=minimum_blob_area,
        )
        self.hue_min_var.set(hue_min)
        self.hue_max_var.set(hue_max)
        self.saturation_min_var.set(saturation_min)
        self.value_min_var.set(value_min)
        self.erosion_iterations_var.set(erosion_iterations)
        self.dilation_iterations_var.set(dilation_iterations)
        self.minimum_blob_area_var.set(minimum_blob_area)

    def _display_latest_frame(self) -> None:
        try:
            error = self.worker.errors.get_nowait()
        except (queue.Empty, OSError, ValueError):
            pass
        else:
            self.status_var.set(f"Capture error: {error}")

        try:
            frame_id, captured_at, frame = self.worker.frames.get_nowait()
        except (queue.Empty, OSError, ValueError):
            pass
        else:
            render_started = time.perf_counter()
            queue_age_ms = (render_started - captured_at) * 1000
            frame = frame.convert("RGB")
            detection_width, detection_height = self.latest_blob_detection_size
            if detection_width > 0 and detection_height > 0:
                scale_x = frame.width / detection_width
                scale_y = frame.height / detection_height
                draw = ImageDraw.Draw(frame)
                outline_width = max(2, round(min(frame.width, frame.height) / 350))
                for left, top, right, bottom in self.latest_blob_boxes:
                    draw.rectangle(
                        (
                            round(left * scale_x),
                            round(top * scale_y),
                            round(right * scale_x),
                            round(bottom * scale_y),
                        ),
                        outline=(255, 64, 64),
                        width=outline_width,
                    )
            preview_width = max(1, self.preview.winfo_width() - 10)
            preview_height = max(1, self.preview.winfo_height() - 10)
            frame.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(frame)
            self.preview.configure(image=self.preview_photo, text="")
            timing_log(
                "UI",
                "preview_rendered",
                frame=frame_id,
                queue_age_ms=f"{queue_age_ms:.1f}",
                render_ms=f"{(time.perf_counter() - render_started) * 1000:.1f}",
                size=f"{frame.width}x{frame.height}",
            )
        self.after(15, self._display_latest_frame)

    def _poll_ocr(self) -> None:
        try:
            while True:
                self.ocr_status_var.set(self.ocr_worker.messages.get_nowait())
        except (queue.Empty, OSError, ValueError):
            pass

        try:
            while True:
                self.ocr_status_var.set(self.ocr_worker.errors.get_nowait())
        except (queue.Empty, OSError, ValueError):
            pass

        latest_update = None
        try:
            while True:
                latest_update = self.ocr_worker.updates.get_nowait()
        except (queue.Empty, OSError, ValueError):
            pass
        if latest_update is not None:
            self.latest_blob_boxes = latest_update.blob_boxes
            self.latest_blob_detection_size = latest_update.detection_size
            words = ", ".join(latest_update.words) or "none"
            stage_timing = ""
            if latest_update.detection_ms is not None:
                stage_timing += f" • det {latest_update.detection_ms:.0f} ms"
            if latest_update.recognition_ms is not None:
                stage_timing += f" • rec {latest_update.recognition_ms:.0f} ms"
            typing_state = ""
            if self.auto_type_value and not is_window_focused(self.target_window):
                typing_state = " • typing paused: selected window is not focused"
            self.ocr_status_var.set(
                f"{latest_update.backend} {latest_update.inference_ms:.0f} ms"
                f"{stage_timing} • "
                f"queue {latest_update.queue_ms:.0f} ms • "
                f"end-to-end {latest_update.capture_to_ocr_ms:.0f} ms • "
                f"detected: {words}"
                f"{typing_state}"
            )
            timing_log(
                "UI",
                "ocr_update_displayed",
                frame=latest_update.frame_id,
                backend=latest_update.backend,
                inference_ms=f"{latest_update.inference_ms:.1f}",
                queue_ms=f"{latest_update.queue_ms:.1f}",
                capture_to_ocr_ms=f"{latest_update.capture_to_ocr_ms:.1f}",
            )
        self.after(50, self._poll_ocr)

    def _stop_capture(self) -> None:
        self.worker.stop()
        self.status_var.set("Capture stopped. Selected area retained; press Start.")
        self.ocr_status_var.set("OCR idle; model remains loaded")

    def _remember_window_geometry(self, _event=None) -> None:
        try:
            if self.state() == "normal":
                self.last_normal_geometry = self.geometry()
        except tk.TclError:
            pass

    def _save_state(self) -> None:
        self._apply_fps()
        self._apply_scale()
        self._apply_recognition_scale()
        self._apply_resize_method()
        self._apply_recognition_resize_method()
        self._apply_crop_padding()
        self._apply_filter()
        self._apply_ocr_settings()
        region = None
        if self.region:
            region = {
                "left": self.region.left,
                "top": self.region.top,
                "width": self.region.width,
                "height": self.region.height,
            }
        data = {
            "window_geometry": self.last_normal_geometry,
            "window_state": self.state(),
            "capture_region": region,
            "target_window": self.target_window if is_window_valid(self.target_window) else None,
            "fps": self.fps_value,
            "scale_percent": self.scale_value,
            "recognition_scale_percent": self.recognition_scale_value,
            "recognition_resize_method": self.recognition_resize_method_value,
            "crop_padding_percent": self.crop_padding_percent_value,
            "resize_method": self.resize_method_value,
            "confidence": self.confidence_value,
            "auto_type": self.auto_type_value,
            "skip_ocr_detector": self.skip_ocr_detector_value,
            "ocr_backend": self.ocr_backend_value,
            "keystroke_delay_ms": self.keystroke_delay_ms,
            "green_filter": {
                "enabled": self.green_filter.enabled,
                "keep_text_bands": self.green_filter.keep_text_bands,
                "hue_min": self.green_filter.hue_min,
                "hue_max": self.green_filter.hue_max,
                "saturation_min": self.green_filter.saturation_min,
                "value_min": self.green_filter.value_min,
                "erosion_iterations": self.green_filter.erosion_iterations,
                "dilation_iterations": self.green_filter.dilation_iterations,
                "minimum_blob_area": self.green_filter.minimum_blob_area,
            },
        }
        temporary = self.state_path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            pass

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        timing_log("MAIN", "shutdown_start")
        self._save_state()
        timing_log("MAIN", "state_saved")
        self.worker.stop()
        self.ocr_worker.stop()
        timing_log("MAIN", "shutdown_complete")
        self.destroy()
