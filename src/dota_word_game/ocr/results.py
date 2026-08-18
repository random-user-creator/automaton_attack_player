from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class OCRUpdate:
    frame_id: int
    backend: str
    words: tuple[str, ...]
    inference_ms: float
    queue_ms: float
    capture_to_ocr_ms: float
    detection_ms: float | None = None
    recognition_ms: float | None = None
    crop_ms: float | None = None
    blob_boxes: tuple[tuple[int, int, int, int], ...] = ()
    detection_size: tuple[int, int] = (0, 0)


def extract_words(output: Any, minimum_confidence: float) -> list[str]:
    detected: list[str] = []
    for item in output or []:
        payload: Any = getattr(item, "json", item)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            continue
        data = payload.get("res", payload)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [1.0] * len(texts))
        for text, score in zip(texts, scores):
            if float(score) >= minimum_confidence:
                detected.extend(re.findall(r"[A-Za-z]{2,}", str(text)))

    unique: list[str] = []
    seen: set[str] = set()
    for word in detected:
        key = word.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(word)
    return unique


def result_dict(item: Any) -> dict[str, Any]:
    payload: Any = getattr(item, "json", item)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return {}
    data = payload.get("res", payload)
    return data if isinstance(data, dict) else {}


def extract_rapid_words(output: Any, minimum_confidence: float) -> list[str]:
    texts = getattr(output, "txts", ()) or ()
    scores = getattr(output, "scores", ()) or ()
    detected: list[str] = []
    for text, score in zip(texts, scores):
        if float(score) >= minimum_confidence:
            detected.extend(re.findall(r"[A-Za-z]{2,}", str(text)))

    unique: list[str] = []
    seen: set[str] = set()
    for word in detected:
        key = word.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(word)
    return unique


def unique_letter_words(candidates: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for word in re.findall(r"[A-Za-z]{2,}", str(candidate)):
            key = word.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(word)
    return unique


def extract_easy_words(output: Any, minimum_confidence: float) -> list[str]:
    candidates: list[str] = []
    for item in output or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        _box, text, score = item[:3]
        if float(score) >= minimum_confidence:
            candidates.append(str(text))
    return unique_letter_words(candidates)


def extract_tesseract_words(output: dict[str, list], minimum_confidence: float) -> list[str]:
    candidates: list[str] = []
    for text, score in zip(output.get("text", []), output.get("conf", [])):
        try:
            confidence = float(score) / 100.0
        except (TypeError, ValueError):
            continue
        if confidence >= minimum_confidence:
            candidates.append(str(text))
    return unique_letter_words(candidates)


def box_bounds(boxes: Any) -> list[tuple[float, float, float, float]]:
    """Normalize polygon or xyxy detections to axis-aligned low-res bounds."""
    normalized: list[tuple[float, float, float, float]] = []
    if boxes is None:
        return normalized
    for box in boxes:
        try:
            values = list(box)
            if len(values) == 4 and all(
                isinstance(value, (int, float)) for value in values
            ):
                left, right, top, bottom = map(float, values)
            else:
                points = [list(point) for point in values]
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        except (TypeError, ValueError, IndexError):
            continue
        if right > left and bottom > top:
            normalized.append((left, top, right, bottom))
    return normalized
