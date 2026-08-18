from __future__ import annotations

from typing import Any

from PIL import Image

from ..vision.processing import GreenFilter

def recognition_crops(
    source: Image.Image,
    detection_size: tuple[int, int],
    bounds: list[tuple[float, float, float, float]],
    recognition_scale_percent: int = 100,
    recognition_resize_method: str = "bicubic",
    crop_padding_percent: float = 5.0,
) -> list[Image.Image]:
    """Map detections to native photo crops, then optionally upscale for REC."""
    detection_width, detection_height = detection_size
    if detection_width <= 0 or detection_height <= 0:
        return []
    scale_x = source.width / detection_width
    scale_y = source.height / detection_height
    crops: list[Image.Image] = []
    for left, top, right, bottom in bounds:
        # Expand every side by the configured fraction before recognition so
        # low-resolution grouping and coordinate rounding cannot clip strokes.
        padding = max(0.0, min(100.0, float(crop_padding_percent))) / 100.0
        pad_x = round((right - left) * scale_x * padding)
        pad_y = round((bottom - top) * scale_y * padding)
        source_box = (
            max(0, int(left * scale_x) - pad_x),
            max(0, int(top * scale_y) - pad_y),
            min(source.width, int(right * scale_x + 0.999) + pad_x),
            min(source.height, int(bottom * scale_y + 0.999) + pad_y),
        )
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            continue
        crop = source.crop(source_box)
        recognition_scale_percent = max(
            100, min(400, int(recognition_scale_percent))
        )
        if recognition_scale_percent > 100:
            resampling = {
                "nearest": Image.Resampling.NEAREST,
                "box": Image.Resampling.BOX,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }.get(str(recognition_resize_method).lower(), Image.Resampling.BICUBIC)
            crop = crop.resize(
                (
                    max(1, round(crop.width * recognition_scale_percent / 100)),
                    max(1, round(crop.height * recognition_scale_percent / 100)),
                ),
                resampling,
            )
        crops.append(crop)
    return crops


def compact_detection_mask(
    frame: Image.Image,
    filter_settings: GreenFilter,
) -> tuple[
    Image.Image,
    list[tuple[int, int, int, int, int, int]] | None,
    int,
]:
    """Pack nearby mask blobs into a small canvas for one detector pass.

    Placements contain packed x/y, source x/y, width and height. ``None`` means
    the unfiltered frame is used directly and needs no coordinate remapping.
    """
    if not filter_settings.enabled:
        return frame, None, 1

    import numpy as np

    # The OCR detection image is inverted: text is black on a white field.
    gray = np.asarray(frame.convert("L"))
    foreground = gray < 128
    active_rows = np.flatnonzero(np.any(foreground, axis=1))
    if active_rows.size == 0:
        return Image.new("RGB", (32, 32), "white"), [], 0

    def grouped_runs(indices, allowed_gap: int) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start = previous = int(indices[0])
        for raw_index in indices[1:]:
            index = int(raw_index)
            if index - previous > allowed_gap + 1:
                runs.append((start, previous + 1))
                start = index
            previous = index
        runs.append((start, previous + 1))
        return runs

    regions: list[tuple[int, int, int, int]] = []
    for band_top, band_bottom in grouped_runs(active_rows, allowed_gap=3):
        band_height = band_bottom - band_top
        active_columns = np.flatnonzero(
            np.any(foreground[band_top:band_bottom, :], axis=0)
        )
        if active_columns.size == 0:
            continue
        letter_gap = max(4, round(band_height * 0.65))
        for run_left, run_right in grouped_runs(active_columns, letter_gap):
            foreground_area = int(
                np.count_nonzero(
                    foreground[band_top:band_bottom, run_left:run_right]
                )
            )
            if foreground_area < filter_settings.minimum_blob_area:
                continue
            left = max(0, run_left - 3)
            top = max(0, band_top - 3)
            right = min(frame.width, run_right + 3)
            bottom = min(frame.height, band_bottom + 3)
            if right - left >= 4 and bottom - top >= 4:
                regions.append((left, top, right, bottom))
    if not regions:
        return Image.new("RGB", (32, 32), "white"), [], 0
    regions.sort(key=lambda box: (box[1], box[0]))

    outer_padding = 8
    packed_width = max(right - left for left, top, right, bottom in regions)
    packed_width += outer_padding * 2
    packed_height = sum(bottom - top for left, top, right, bottom in regions)
    packed_height += outer_padding * (len(regions) + 1)
    packed = Image.new("RGB", (max(32, packed_width), max(32, packed_height)), "white")
    placements: list[tuple[int, int, int, int, int, int]] = []
    packed_y = outer_padding
    for left, top, right, bottom in regions:
        width, height = right - left, bottom - top
        packed_x = outer_padding
        packed.paste(frame.crop((left, top, right, bottom)), (packed_x, packed_y))
        placements.append((packed_x, packed_y, left, top, width, height))
        packed_y += height + outer_padding
    return packed, placements, len(regions)


def unpack_detection_bounds(
    bounds: list[tuple[float, float, float, float]],
    placements: list[tuple[int, int, int, int, int, int]] | None,
) -> list[tuple[float, float, float, float]]:
    if placements is None:
        return bounds
    unpacked: list[tuple[float, float, float, float]] = []
    for left, top, right, bottom in bounds:
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        for packed_x, packed_y, source_x, source_y, width, height in placements:
            if (
                packed_x <= center_x <= packed_x + width
                and packed_y <= center_y <= packed_y + height
            ):
                unpacked.append(
                    (
                        max(source_x, source_x + left - packed_x),
                        max(source_y, source_y + top - packed_y),
                        min(source_x + width, source_x + right - packed_x),
                        min(source_y + height, source_y + bottom - packed_y),
                    )
                )
                break
    return [box for box in unpacked if box[2] > box[0] and box[3] > box[1]]
