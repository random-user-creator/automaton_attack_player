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
