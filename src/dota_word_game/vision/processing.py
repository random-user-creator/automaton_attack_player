from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageFilter


RESIZE_METHODS = ("nearest", "box", "bilinear", "lanczos")


@dataclass(frozen=True)
class GreenFilter:
    enabled: bool = True
    keep_text_bands: bool = False
    hue_min: int = 80
    hue_max: int = 93
    saturation_min: int = 38
    value_min: int = 50
    erosion_iterations: int = 0
    dilation_iterations: int = 0
    minimum_blob_area: int = 12


def isolate_green_text(frame: Image.Image, settings: GreenFilter) -> Image.Image:
    """Return white matching pixels on black for OCR-friendly green text."""
    if not settings.enabled:
        return frame

    hue, saturation, value = frame.convert("HSV").split()
    hue_min = round(settings.hue_min * 255 / 360)
    hue_max = round(settings.hue_max * 255 / 360)
    saturation_min = round(settings.saturation_min * 255 / 100)
    value_min = round(settings.value_min * 255 / 100)

    hue_mask = hue.point(lambda pixel: 255 if hue_min <= pixel <= hue_max else 0)
    saturation_mask = saturation.point(
        lambda pixel: 255 if pixel >= saturation_min else 0
    )
    value_mask = value.point(lambda pixel: 255 if pixel >= value_min else 0)
    mask = ImageChops.multiply(ImageChops.multiply(hue_mask, saturation_mask), value_mask)
    for _ in range(settings.erosion_iterations):
        mask = mask.filter(ImageFilter.MinFilter(3))
    for _ in range(settings.dilation_iterations):
        mask = mask.filter(ImageFilter.MaxFilter(3))
    if settings.keep_text_bands:
        mask = keep_text_bands(mask)
    return Image.merge("RGB", (mask, mask, mask))


def keep_text_bands(mask: Image.Image) -> Image.Image:
    """Keep multiple letter-like horizontal bands and reject scenery blobs."""
    width, height = mask.size
    if width == 0 or height == 0:
        return mask

    row_strengths = mask.resize((1, height), Image.Resampling.BOX).tobytes()
    peak_strength = max(row_strengths)
    if peak_strength == 0:
        return mask

    active_threshold = max(1, round(peak_strength * 0.05))
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last_active = 0
    gap = 0
    for row, strength in enumerate(row_strengths):
        if strength >= active_threshold:
            if start is None:
                start = row
            last_active = row
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                bands.append((start, last_active))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, last_active))

    cleaned = Image.new("L", mask.size)
    minimum_span = max(8, round(width * 0.04))
    for top, bottom in bands:
        band = mask.crop((0, top, width, bottom + 1))
        column_strengths = band.resize((width, 1), Image.Resampling.BOX).tobytes()
        occupied = [column for column, strength in enumerate(column_strengths) if strength]
        if not occupied:
            continue

        span = occupied[-1] - occupied[0] + 1
        runs = 0
        previous = -2
        for column in occupied:
            if column != previous + 1:
                runs += 1
            previous = column

        white_pixels = band.histogram()[255]
        density = white_pixels / (span * band.height)
        if span < minimum_span or runs < 2 or not 0.20 <= density <= 0.72:
            continue

        paste_top = max(0, top - 2)
        paste_bottom = min(height, bottom + 3)
        cleaned.paste(mask.crop((0, paste_top, width, paste_bottom)), (0, paste_top))
    return cleaned


def scale_for_processing(frame: Image.Image, percentage: int) -> Image.Image:
    """Downscale a captured frame before filtering and detection."""
    percentage = max(10, min(100, percentage))
    if percentage == 100:
        return frame
    width = max(1, round(frame.width * percentage / 100))
    height = max(1, round(frame.height * percentage / 100))
    return frame.resize((width, height), Image.Resampling.LANCZOS)


def normalize_resize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    return normalized if normalized in RESIZE_METHODS else "nearest"


def resize_rgb_array(frame: Any, percentage: int, method: str = "nearest"):
    """Resize an RGB ndarray, using a pure NumPy fast path for nearest."""
    import numpy as np

    source = np.asarray(frame, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("Expected an RGB image array.")
    source = source[:, :, :3]
    percentage = max(10, min(100, int(percentage)))
    method = normalize_resize_method(method)
    if percentage == 100:
        return source

    source_height, source_width = source.shape[:2]
    width = max(1, round(source_width * percentage / 100))
    height = max(1, round(source_height * percentage / 100))
    if method == "nearest":
        # Integer lookup avoids Pillow conversion and is considerably faster
        # than any interpolating resize. Advanced indexing creates the owned,
        # compact output required by the following native operations.
        rows = np.minimum(
            (np.arange(height, dtype=np.int64) * source_height) // height,
            source_height - 1,
        )
        columns = np.minimum(
            (np.arange(width, dtype=np.int64) * source_width) // width,
            source_width - 1,
        )
        return np.ascontiguousarray(source[rows[:, None], columns[None, :]])

    try:
        import cv2

        interpolation = {
            "box": cv2.INTER_AREA,
            "bilinear": cv2.INTER_LINEAR,
            "lanczos": cv2.INTER_LANCZOS4,
        }[method]
        return cv2.resize(source, (width, height), interpolation=interpolation)
    except ImportError:
        resampling = {
            "box": Image.Resampling.BOX,
            "bilinear": Image.Resampling.BILINEAR,
            "lanczos": Image.Resampling.LANCZOS,
        }[method]
        return np.asarray(
            Image.fromarray(source, "RGB").resize((width, height), resampling)
        )


def _keep_text_bands_array(mask):
    """NumPy equivalent of keep_text_bands for a uint8 white mask."""
    import numpy as np

    height, width = mask.shape
    if width == 0 or height == 0:
        return mask
    row_strengths = mask.mean(axis=1)
    peak_strength = float(row_strengths.max(initial=0.0))
    if peak_strength == 0:
        return mask

    active_threshold = max(1.0, round(peak_strength * 0.05))
    active_rows = row_strengths >= active_threshold
    bands: list[tuple[int, int]] = []
    start: int | None = None
    last_active = 0
    gap = 0
    for row, active in enumerate(active_rows):
        if active:
            if start is None:
                start = row
            last_active = row
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                bands.append((start, last_active))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, last_active))

    cleaned = np.zeros_like(mask)
    minimum_span = max(8, round(width * 0.04))
    for top, bottom in bands:
        band = mask[top : bottom + 1]
        occupied = np.flatnonzero(np.any(band != 0, axis=0))
        if occupied.size == 0:
            continue
        span = int(occupied[-1] - occupied[0] + 1)
        runs = 1 + int(np.count_nonzero(np.diff(occupied) != 1))
        white_pixels = int(np.count_nonzero(band))
        density = white_pixels / (span * band.shape[0])
        if span < minimum_span or runs < 2 or not 0.20 <= density <= 0.72:
            continue
        paste_top = max(0, top - 2)
        paste_bottom = min(height, bottom + 3)
        cleaned[paste_top:paste_bottom] = mask[paste_top:paste_bottom]
    return cleaned


def isolate_green_text_array(frame, settings: GreenFilter):
    """Produce a single-channel mask with native ndarray operations."""
    import numpy as np

    source = np.asarray(frame, dtype=np.uint8)
    if not settings.enabled:
        return source
    try:
        import cv2

        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        hue_min = max(0, min(179, round(settings.hue_min / 2)))
        hue_max = max(0, min(179, round(settings.hue_max / 2)))
        saturation_min = round(settings.saturation_min * 255 / 100)
        value_min = round(settings.value_min * 255 / 100)
        mask = cv2.inRange(
            hsv,
            np.array((hue_min, saturation_min, value_min), dtype=np.uint8),
            np.array((hue_max, 255, 255), dtype=np.uint8),
        )
        if settings.erosion_iterations:
            mask = cv2.erode(
                mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=settings.erosion_iterations,
            )
        if settings.dilation_iterations:
            mask = cv2.dilate(
                mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=settings.dilation_iterations,
            )
        if settings.keep_text_bands:
            mask = _keep_text_bands_array(mask)
        return mask
    except ImportError:
        # Keep the app usable in a minimal environment; the normal project
        # install provides OpenCV through EasyOCR.
        return np.asarray(
            isolate_green_text(Image.fromarray(source, "RGB"), settings).convert("L")
        )


def process_detection_array(
    source,
    percentage: int,
    resize_method: str,
    settings: GreenFilter,
) -> Image.Image:
    """Resize and filter an RGB ndarray, returning an OCR/preview image."""
    processed = resize_rgb_array(source, percentage, resize_method)
    filtered = isolate_green_text_array(processed, settings)
    if settings.enabled:
        return Image.fromarray(filtered, "L")
    return Image.fromarray(filtered, "RGB")
