import sys


RESIZE_METHOD_LABELS = {
    "nearest": "Nearest (fastest)",
    "box": "Box",
    "bilinear": "Bilinear",
    "lanczos": "Lanczos",
}
RESIZE_LABEL_METHODS = {
    label: method for method, label in RESIZE_METHOD_LABELS.items()
}

REC_RESIZE_METHOD_LABELS = {
    "nearest": "Nearest",
    "box": "Box",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "lanczos": "Lanczos",
}
REC_RESIZE_LABEL_METHODS = {
    label: method for method, label in REC_RESIZE_METHOD_LABELS.items()
}

OCR_BACKEND_LABELS = {
    "rapidocr": "RapidOCR CPU",
    "rapidocr_gpu": "RapidOCR GPU (CUDA)",
    "paddle_cpu": "PaddleOCR CPU",
    "paddle": "PaddleOCR GPU",
    "paddle_server_cpu": "PaddleOCR Server CPU",
    "paddle_server_gpu": "PaddleOCR Server GPU",
    "easyocr": "EasyOCR CPU",
    "easyocr_gpu": "EasyOCR GPU (CUDA)",
    "tesseract": "Tesseract CPU",
}
OCR_LABEL_BACKENDS = {
    label: backend for backend, label in OCR_BACKEND_LABELS.items()
}
SOURCE_OCR_BACKENDS = tuple(OCR_BACKEND_LABELS)
PACKAGED_OCR_BACKENDS = ("paddle_cpu",)


def available_ocr_backends() -> tuple[str, ...]:
    """The one-file release intentionally bundles one offline CPU backend."""
    return (
        PACKAGED_OCR_BACKENDS
        if getattr(sys, "frozen", False)
        else SOURCE_OCR_BACKENDS
    )
