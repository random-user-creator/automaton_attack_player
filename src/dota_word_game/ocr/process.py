from __future__ import annotations

import queue
import shutil
import signal
import time

from PIL import Image

from ..logging import timing_log
from ..paths import bundled_model_dir
from ..queueing import put_latest
from .crops import compact_detection_mask, recognition_crops, unpack_detection_bounds
from .results import (
    OCRUpdate,
    box_bounds,
    extract_easy_words,
    extract_rapid_words,
    extract_tesseract_words,
    extract_words,
    result_dict,
    unique_letter_words,
)

PADDLE_BACKENDS = {
    "paddle_cpu": {
        "device": "cpu",
        "detector": "PP-OCRv4_mobile_det",
        "recognizer": "PP-OCRv4_mobile_rec",
        "profile": "mobile",
    },
    "paddle": {
        "device": "gpu:0",
        "detector": "PP-OCRv4_mobile_det",
        "recognizer": "PP-OCRv4_mobile_rec",
        "profile": "mobile",
    },
    "paddle_server_cpu": {
        "device": "cpu",
        "detector": "PP-OCRv5_server_det",
        "recognizer": "PP-OCRv5_server_rec",
        "profile": "server",
    },
    "paddle_server_gpu": {
        "device": "gpu:0",
        "detector": "PP-OCRv5_server_det",
        "recognizer": "PP-OCRv5_server_rec",
        "profile": "server",
    },
}

# PaddleOCR defaults to ten inference threads. That oversubscribes this live
# pipeline and competes with capture, filtering, the UI, and keyboard output.
# Small word crops benchmark faster with two threads on the target machine.
PADDLE_CPU_THREADS = 2


def ocr_process_main(
    backend,
    stop_event,
    frames,
    typing_commands,
    updates,
    messages,
    errors,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    load_started = time.perf_counter()
    timing_log("OCR", "model_load_start", backend=backend)
    try:
        import numpy as np

        if backend in {"rapidocr", "rapidocr_gpu"}:
            import onnxruntime as ort
            from rapidocr import (
                EngineType,
                LangDet,
                LangRec,
                ModelType,
                OCRVersion,
                RapidOCR,
            )

            rapid_gpu = backend == "rapidocr_gpu"
            if rapid_gpu:
                preload_dlls = getattr(ort, "preload_dlls", None)
                if callable(preload_dlls):
                    preload_dlls()
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" not in available_providers:
                    raise RuntimeError(
                        "RapidOCR GPU requested, but ONNX Runtime has no "
                        "CUDAExecutionProvider. Run setup_rapidocr_gpu.ps1. "
                        f"Available providers: {available_providers}"
                    )
            put_latest(
                messages,
                "Loading RapidOCR with ONNX Runtime "
                f"{'CUDA' if rapid_gpu else 'CPU'}…",
            )
            rapid_params = {
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.EN,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.EN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
                "EngineConfig.onnxruntime.use_cuda": rapid_gpu,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
            }
            if rapid_gpu:
                rapid_params[
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"
                ] = "HEURISTIC"
            ocr = RapidOCR(
                params=rapid_params
            )
            if rapid_gpu:
                stage_providers = {
                    "det": ocr.text_det.session.session.get_providers(),
                    "rec": ocr.text_rec.session.session.get_providers(),
                }
                non_cuda = {
                    stage: providers
                    for stage, providers in stage_providers.items()
                    if not providers or providers[0] != "CUDAExecutionProvider"
                }
                if non_cuda:
                    raise RuntimeError(
                        "RapidOCR GPU initialized without CUDA as the primary "
                        f"provider: {non_cuda}"
                    )
                device = "ONNX Runtime CUDA GPU 0"
                timing_log(
                    "OCR",
                    "cuda_provider_config",
                    cudnn_conv_algo_search="HEURISTIC",
                    providers=stage_providers,
                )
            else:
                device = "ONNX Runtime CPU"
        elif backend in PADDLE_BACKENDS:
            paddle_settings = PADDLE_BACKENDS[backend]
            paddle_device = paddle_settings["device"]
            paddle_gpu = paddle_device.startswith("gpu")
            paddle_profile = paddle_settings["profile"]
            put_latest(
                messages,
                f"Loading PaddleOCR {paddle_profile} models on "
                f"{'GPU' if paddle_gpu else 'CPU'}…",
            )
            from paddleocr import TextDetection, TextRecognition
            import paddle

            if paddle_gpu and not paddle.is_compiled_with_cuda():
                raise RuntimeError(
                    "PaddleOCR GPU requested, but PaddlePaddle has no CUDA "
                    "support. Run scripts/setup_paddleocr.ps1."
                )
            detector_name = paddle_settings["detector"]
            recognizer_name = paddle_settings["recognizer"]
            detector_dir = bundled_model_dir(detector_name)
            recognizer_dir = bundled_model_dir(recognizer_name)
            detector_kwargs = {
                "model_name": detector_name,
                "device": paddle_device,
            }
            recognizer_kwargs = {
                "model_name": recognizer_name,
                "device": paddle_device,
            }
            if not paddle_gpu:
                detector_kwargs["cpu_threads"] = PADDLE_CPU_THREADS
                recognizer_kwargs["cpu_threads"] = PADDLE_CPU_THREADS
            if detector_dir is not None:
                detector_kwargs["model_dir"] = str(detector_dir)
            if recognizer_dir is not None:
                recognizer_kwargs["model_dir"] = str(recognizer_dir)
            detector = TextDetection(**detector_kwargs)
            recognizer = TextRecognition(**recognizer_kwargs)
            ocr = (detector, recognizer)
            device = f"PaddlePaddle {paddle_device} ({paddle_profile})"
        elif backend in {"easyocr", "easyocr_gpu"}:
            import easyocr
            import torch

            easy_gpu = backend == "easyocr_gpu"
            if easy_gpu and not torch.cuda.is_available():
                raise RuntimeError(
                    "EasyOCR GPU requested, but PyTorch CUDA is unavailable. "
                    "Run setup_extra_ocr.ps1."
                )
            put_latest(
                messages,
                f"Loading EasyOCR on {'CUDA' if easy_gpu else 'CPU'}…",
            )
            # Dynamic screen and word-crop sizes make cuDNN benchmarking prone
            # to latency spikes, so keep it disabled for this live workload.
            ocr = easyocr.Reader(
                ["en"],
                gpu=easy_gpu,
                verbose=False,
                cudnn_benchmark=False,
            )
            easy_device = str(getattr(ocr, "device", "unknown"))
            if easy_gpu and not easy_device.startswith("cuda"):
                raise RuntimeError(
                    f"EasyOCR requested CUDA but initialized on {easy_device}."
                )
            device = (
                f"PyTorch CUDA: {torch.cuda.get_device_name(0)}"
                if easy_gpu
                else "PyTorch CPU"
            )
        elif backend == "tesseract":
            import os

            import pytesseract

            tesseract_command = shutil.which("tesseract")
            if not tesseract_command:
                for environment_name in ("ProgramFiles", "ProgramFiles(x86)"):
                    program_files = os.environ.get(environment_name)
                    if not program_files:
                        continue
                    candidate = os.path.join(
                        program_files, "Tesseract-OCR", "tesseract.exe"
                    )
                    if os.path.isfile(candidate):
                        tesseract_command = candidate
                        break
            if not tesseract_command:
                raise RuntimeError(
                    "Tesseract executable was not found. Install Tesseract 5 "
                    "and add it to PATH, then restart the app."
                )
            pytesseract.pytesseract.tesseract_cmd = tesseract_command
            version = pytesseract.get_tesseract_version()
            ocr = pytesseract
            device = f"Tesseract {version} CPU"
        else:
            raise RuntimeError(f"Unknown OCR backend: {backend}")
        load_ms = (time.perf_counter() - load_started) * 1000
        put_latest(messages, f"{backend} ready on {device}")
        timing_log(
            "OCR",
            "model_load_complete",
            backend=backend,
            device=device,
            load_ms=f"{load_ms:.1f}",
        )
    except Exception as exc:
        timing_log("OCR", "model_load_failed", error=repr(exc))
        put_latest(errors, f"OCR initialization failed: {exc}")
        return

    while not stop_event.is_set():
        try:
            (
                frame_id,
                captured_at,
                submitted_at,
                detection_mode,
                detection_size,
                detection_pixels,
                source_mode,
                source_size,
                source_pixels,
                filter_settings,
                confidence,
                auto_type,
                target_window,
                keystroke_delay_ms,
                recognition_scale_percent,
                recognition_resize_method,
                skip_ocr_detector,
                crop_padding_percent,
            ) = frames.get(timeout=0.1)
        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError, OSError):
            return

        dequeued_at = time.perf_counter()
        queue_ms = (dequeued_at - submitted_at) * 1000
        frame_age_ms = (dequeued_at - captured_at) * 1000
        decode_started = time.perf_counter()
        frame = Image.frombytes(detection_mode, detection_size, detection_pixels)
        source_frame = Image.frombytes(source_mode, source_size, source_pixels)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        started = time.perf_counter()
        compact_started = time.perf_counter()
        detector_frame, packed_placements, blob_groups = compact_detection_mask(
            frame, filter_settings
        )
        blob_boxes = (
            tuple(
                (source_x, source_y, source_x + width, source_y + height)
                for _packed_x, _packed_y, source_x, source_y, width, height in packed_placements
            )
            if packed_placements is not None
            else ()
        )
        compact_ms = (time.perf_counter() - compact_started) * 1000
        skip_detector_active = bool(
            skip_ocr_detector and packed_placements is not None
        )
        direct_bounds = (
            [
                (source_x, source_y, source_x + width, source_y + height)
                for _packed_x, _packed_y, source_x, source_y, width, height
                in packed_placements
            ]
            if skip_detector_active
            else []
        )
        detection_ms = None
        recognition_ms = None
        crop_ms = None
        crop_count = 0
        if blob_groups == 0:
            # The compacting stage has already proven that the filtered mask
            # contains no candidate text. Avoid paying the detector's fixed
            # cost on the otherwise blank 32x32 placeholder image.
            words: list[str] = []
            detection_ms = 0.0
            recognition_ms = 0.0
            crop_ms = 0.0
            finished_at = time.perf_counter()
            inference_ms = (finished_at - started) * 1000
            capture_to_ocr_ms = (finished_at - captured_at) * 1000
            timing_log(
                "OCR",
                "frame_complete",
                frame=frame_id,
                backend=backend,
                queue_ms=f"{queue_ms:.1f}",
                frame_age_ms=f"{frame_age_ms:.1f}",
                decode_ms=f"{decode_ms:.1f}",
                inference_ms=f"{inference_ms:.1f}",
                capture_to_ocr_ms=f"{capture_to_ocr_ms:.1f}",
                detection_ms="0.0",
                recognition_ms="0.0",
                crop_ms="0.0",
                compact_ms=f"{compact_ms:.1f}",
                blob_groups=0,
                crops=0,
                detection_size=f"{frame.width}x{frame.height}",
                detector_input_size="skipped",
                source_size=f"{source_frame.width}x{source_frame.height}",
                recognition_scale_percent=recognition_scale_percent,
                recognition_resize_method=recognition_resize_method,
                detector_skipped=True,
                crop_padding_percent=crop_padding_percent,
                words=0,
                skip_reason="no_blobs",
            )
            put_latest(
                updates,
                OCRUpdate(
                    frame_id,
                    backend,
                    (),
                    inference_ms,
                    queue_ms,
                    capture_to_ocr_ms,
                    detection_ms,
                    recognition_ms,
                    crop_ms,
                    blob_boxes,
                    frame.size,
                ),
            )
            continue
        try:
            if backend in {"rapidocr", "rapidocr_gpu"}:
                if skip_detector_active:
                    detection_ms = 0.0
                    bounds = direct_bounds
                else:
                    detection_started = time.perf_counter()
                    detection = ocr.text_det(np.asarray(detector_frame))
                    detection_ms = (time.perf_counter() - detection_started) * 1000
                    bounds = unpack_detection_bounds(
                        box_bounds(getattr(detection, "boxes", None)),
                        packed_placements,
                    )
                crop_started = time.perf_counter()
                crops = recognition_crops(
                    source_frame,
                    frame.size,
                    bounds,
                    recognition_scale_percent,
                    recognition_resize_method,
                    crop_padding_percent,
                )
                crop_ms = (time.perf_counter() - crop_started) * 1000
                crop_count = len(crops)
                recognition_started = time.perf_counter()
                if crops:
                    recognition = ocr.recognize_txt(
                        [np.asarray(crop) for crop in crops]
                    )
                    words = extract_rapid_words(recognition, confidence)
                else:
                    words = []
                recognition_ms = (time.perf_counter() - recognition_started) * 1000
            elif backend in PADDLE_BACKENDS:
                detector, recognizer = ocr
                if skip_detector_active:
                    detection_ms = 0.0
                    bounds = direct_bounds
                else:
                    detection_started = time.perf_counter()
                    detection_output = detector.predict(np.asarray(detector_frame))
                    detection_ms = (time.perf_counter() - detection_started) * 1000
                    detection_data = (
                        result_dict(detection_output[0]) if detection_output else {}
                    )
                    bounds = unpack_detection_bounds(
                        box_bounds(detection_data.get("dt_polys")), packed_placements
                    )
                crop_started = time.perf_counter()
                crops = recognition_crops(
                    source_frame,
                    frame.size,
                    bounds,
                    recognition_scale_percent,
                    recognition_resize_method,
                    crop_padding_percent,
                )
                crop_ms = (time.perf_counter() - crop_started) * 1000
                crop_count = len(crops)
                recognition_started = time.perf_counter()
                recognition_output = (
                    recognizer.predict([np.asarray(crop) for crop in crops])
                    if crops
                    else []
                )
                recognition_ms = (time.perf_counter() - recognition_started) * 1000
                candidates: list[str] = []
                for item in recognition_output:
                    data = result_dict(item)
                    try:
                        score = float(data.get("rec_score", 0.0))
                    except (TypeError, ValueError):
                        continue
                    if score >= confidence:
                        candidates.append(str(data.get("rec_text", "")))
                words = unique_letter_words(candidates)
            elif backend in {"easyocr", "easyocr_gpu"}:
                from easyocr.utils import reformat_input

                if skip_detector_active:
                    detection_ms = 0.0
                    bounds = direct_bounds
                else:
                    low_image, _low_gray = reformat_input(np.asarray(detector_frame))
                    detection_started = time.perf_counter()
                    horizontal_agg, free_agg = ocr.detect(low_image)
                    detection_ms = (time.perf_counter() - detection_started) * 1000
                    horizontal = horizontal_agg[0] if horizontal_agg else []
                    free = free_agg[0] if free_agg else []
                    bounds = [
                        (float(box[0]), float(box[2]), float(box[1]), float(box[3]))
                        for box in horizontal
                        if len(box) >= 4
                    ]
                    bounds.extend(box_bounds(free))
                    bounds = unpack_detection_bounds(bounds, packed_placements)
                crop_started = time.perf_counter()
                crops = recognition_crops(
                    source_frame,
                    frame.size,
                    bounds,
                    recognition_scale_percent,
                    recognition_resize_method,
                    crop_padding_percent,
                )
                crop_ms = (time.perf_counter() - crop_started) * 1000
                crop_count = len(crops)
                recognition_started = time.perf_counter()
                easy_results: list[Any] = []
                for crop in crops:
                    _crop_image, crop_gray = reformat_input(np.asarray(crop))
                    easy_results.extend(
                        ocr.recognize(
                            crop_gray,
                            horizontal_list=[[0, crop.width, 0, crop.height]],
                            free_list=[],
                            decoder="greedy",
                            batch_size=1,
                            workers=0,
                            allowlist=(
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                            ),
                        )
                    )
                recognition_ms = (time.perf_counter() - recognition_started) * 1000
                words = extract_easy_words(easy_results, confidence)
            elif backend == "tesseract":
                from pytesseract import Output

                if skip_detector_active:
                    detection_ms = 0.0
                    bounds = direct_bounds
                else:
                    detection_started = time.perf_counter()
                    detection_output = ocr.image_to_data(
                        detector_frame,
                        lang="eng",
                        config=(
                            "--oem 1 --psm 11 "
                            "-c tessedit_char_whitelist="
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                        ),
                        output_type=Output.DICT,
                        timeout=5.0,
                    )
                    detection_ms = (time.perf_counter() - detection_started) * 1000
                    bounds = []
                    for left, top, width, height, score in zip(
                        detection_output.get("left", []),
                        detection_output.get("top", []),
                        detection_output.get("width", []),
                        detection_output.get("height", []),
                        detection_output.get("conf", []),
                    ):
                        try:
                            left = float(left)
                            top = float(top)
                            width = float(width)
                            height = float(height)
                            score = float(score)
                        except (TypeError, ValueError):
                            continue
                        if score >= 0 and width > 0 and height > 0:
                            bounds.append((left, top, left + width, top + height))
                    bounds = unpack_detection_bounds(bounds, packed_placements)
                crop_started = time.perf_counter()
                crops = recognition_crops(
                    source_frame,
                    frame.size,
                    bounds,
                    recognition_scale_percent,
                    recognition_resize_method,
                    crop_padding_percent,
                )
                crop_ms = (time.perf_counter() - crop_started) * 1000
                crop_count = len(crops)
                recognition_started = time.perf_counter()
                candidates: list[str] = []
                for crop in crops:
                    crop_output = ocr.image_to_data(
                        crop,
                        lang="eng",
                        config=(
                            "--oem 1 --psm 7 "
                            "-c tessedit_char_whitelist="
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                        ),
                        output_type=Output.DICT,
                        timeout=5.0,
                    )
                    candidates.extend(
                        extract_tesseract_words(crop_output, confidence)
                    )
                recognition_ms = (time.perf_counter() - recognition_started) * 1000
                words = unique_letter_words(candidates)
            else:
                raise RuntimeError(f"Unknown OCR backend: {backend}")
        except Exception as exc:
            timing_log("OCR", "inference_failed", frame=frame_id, error=repr(exc))
            put_latest(errors, f"OCR inference failed: {exc}")
            continue

        finished_at = time.perf_counter()
        inference_ms = (finished_at - started) * 1000
        capture_to_ocr_ms = (finished_at - captured_at) * 1000
        timing_log(
            "OCR",
            "frame_complete",
            frame=frame_id,
            backend=backend,
            queue_ms=f"{queue_ms:.1f}",
            frame_age_ms=f"{frame_age_ms:.1f}",
            decode_ms=f"{decode_ms:.1f}",
            inference_ms=f"{inference_ms:.1f}",
            capture_to_ocr_ms=f"{capture_to_ocr_ms:.1f}",
            detection_ms=(f"{detection_ms:.1f}" if detection_ms is not None else "n/a"),
            recognition_ms=(
                f"{recognition_ms:.1f}" if recognition_ms is not None else "n/a"
            ),
            crop_ms=(f"{crop_ms:.1f}" if crop_ms is not None else "n/a"),
            compact_ms=f"{compact_ms:.1f}",
            blob_groups=blob_groups,
            crops=crop_count,
            detection_size=f"{frame.width}x{frame.height}",
            detector_input_size=f"{detector_frame.width}x{detector_frame.height}",
            source_size=f"{source_frame.width}x{source_frame.height}",
            recognition_scale_percent=recognition_scale_percent,
            recognition_resize_method=recognition_resize_method,
            detector_skipped=skip_detector_active,
            crop_padding_percent=crop_padding_percent,
            words=len(words),
        )
        put_latest(
            updates,
            OCRUpdate(
                frame_id,
                backend,
                tuple(words),
                inference_ms,
                queue_ms,
                capture_to_ocr_ms,
                detection_ms,
                recognition_ms,
                crop_ms,
                blob_boxes,
                frame.size,
            ),
        )
        if words and auto_type:
            put_latest(
                typing_commands,
                (
                    frame_id,
                    captured_at,
                    finished_at,
                    tuple(words),
                    target_window,
                    keystroke_delay_ms,
                ),
            )
