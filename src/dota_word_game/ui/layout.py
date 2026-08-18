from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .constants import (
    OCR_BACKEND_LABELS,
    REC_RESIZE_LABEL_METHODS,
    REC_RESIZE_METHOD_LABELS,
    RESIZE_LABEL_METHODS,
    RESIZE_METHOD_LABELS,
)


class UILayoutMixin:
    def _build_ui(self) -> None:
        actions = ttk.Frame(self, padding=(12, 12, 12, 6))
        actions.pack(fill="x")
        actions.columnconfigure(0, weight=1)

        ttk.Button(
            actions,
            text="select_area",
            command=self._open_region_selector,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Start", command=self._start_capture).grid(
            row=0, column=1, sticky="e", padx=(6, 4)
        )
        ttk.Button(actions, text="Stop", command=self._stop_capture).grid(
            row=0, column=2, sticky="e"
        )

        capture_controls = ttk.LabelFrame(self, text="Capture settings", padding=8)
        capture_controls.pack(fill="x", padx=12, pady=(0, 8))
        capture_controls.columnconfigure(3, weight=1)

        ttk.Label(capture_controls, text="FPS:").grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        fps_spinbox = ttk.Spinbox(
            capture_controls,
            from_=1,
            to=60,
            width=5,
            textvariable=self.fps_var,
            command=self._apply_fps,
        )
        fps_spinbox.grid(row=0, column=1, sticky="w")
        fps_spinbox.bind("<Return>", self._apply_fps)
        fps_spinbox.bind("<FocusOut>", self._apply_fps)

        ttk.Label(capture_controls, text="Processing resolution:").grid(
            row=0, column=2, sticky="w", padx=(18, 6)
        )
        ttk.Scale(
            capture_controls,
            from_=10,
            to=100,
            variable=self.scale_var,
            command=self._apply_scale,
        ).grid(row=0, column=3, sticky="ew")
        ttk.Label(
            capture_controls,
            textvariable=self.scale_label_var,
            width=5,
        ).grid(row=0, column=4, sticky="e", padx=(5, 0))

        ttk.Label(capture_controls, text="Resize method:").grid(
            row=1, column=0, sticky="w", pady=(7, 0)
        )
        resize_combo = ttk.Combobox(
            capture_controls,
            state="readonly",
            width=18,
            textvariable=self.resize_method_var,
            values=tuple(RESIZE_LABEL_METHODS),
        )
        resize_combo.grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(7, 0),
        )
        resize_combo.bind("<<ComboboxSelected>>", self._apply_resize_method)

        ttk.Label(capture_controls, text="Recognition crop scale:").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(7, 0)
        )
        ttk.Scale(
            capture_controls,
            from_=100,
            to=400,
            variable=self.recognition_scale_var,
            command=self._apply_recognition_scale,
        ).grid(
            row=2,
            column=2,
            columnspan=2,
            sticky="ew",
            padx=(8, 0),
            pady=(7, 0),
        )
        ttk.Label(
            capture_controls,
            textvariable=self.recognition_scale_label_var,
            width=5,
        ).grid(row=2, column=4, sticky="e", padx=(5, 0), pady=(7, 0))

        ttk.Label(capture_controls, text="REC resize method:").grid(
            row=3, column=0, sticky="w", pady=(7, 0)
        )
        rec_resize_combo = ttk.Combobox(
            capture_controls,
            state="readonly",
            width=18,
            textvariable=self.recognition_resize_method_var,
            values=tuple(REC_RESIZE_LABEL_METHODS),
        )
        rec_resize_combo.grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(7, 0),
        )
        rec_resize_combo.bind(
            "<<ComboboxSelected>>", self._apply_recognition_resize_method
        )

        ttk.Label(capture_controls, text="REC crop padding:").grid(
            row=4, column=0, sticky="w", pady=(7, 0)
        )
        crop_padding_spinbox = ttk.Spinbox(
            capture_controls,
            from_=0,
            to=100,
            increment=1,
            width=7,
            textvariable=self.crop_padding_percent_var,
            command=self._apply_crop_padding,
        )
        crop_padding_spinbox.grid(row=4, column=1, sticky="w", pady=(7, 0))
        crop_padding_spinbox.bind("<Return>", self._apply_crop_padding)
        crop_padding_spinbox.bind("<FocusOut>", self._apply_crop_padding)
        ttk.Label(capture_controls, text="% on each side").grid(
            row=4, column=2, sticky="w", padx=(5, 0), pady=(7, 0)
        )

        ocr_controls = ttk.LabelFrame(self, text="OCR and typing", padding=8)
        ocr_controls.pack(fill="x", padx=12, pady=(0, 8))
        ocr_controls.columnconfigure(4, weight=1)
        ttk.Checkbutton(
            ocr_controls,
            text="Auto-type detected words",
            variable=self.auto_type_var,
            command=self._apply_ocr_settings,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(ocr_controls, text="Minimum confidence:").grid(
            row=0, column=1, sticky="w", padx=(18, 5)
        )
        confidence_spinbox = ttk.Spinbox(
            ocr_controls,
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=5,
            textvariable=self.confidence_var,
            command=self._apply_ocr_settings,
        )
        confidence_spinbox.grid(row=0, column=2, sticky="w")
        confidence_spinbox.bind("<Return>", self._apply_ocr_settings)
        confidence_spinbox.bind("<FocusOut>", self._apply_ocr_settings)
        ttk.Checkbutton(
            ocr_controls,
            text="Recognize blob groups directly (skip detector)",
            variable=self.skip_ocr_detector_var,
            command=self._apply_ocr_settings,
        ).grid(row=0, column=3, columnspan=2, sticky="w", padx=(18, 0))
        ttk.Label(ocr_controls, textvariable=self.ocr_status_var).grid(
            row=2,
            column=0,
            columnspan=5,
            sticky="w",
            pady=(6, 0),
        )
        ttk.Label(ocr_controls, text="OCR backend:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        backend_combo = ttk.Combobox(
            ocr_controls,
            state="readonly",
            width=22,
            textvariable=self.ocr_backend_var,
            values=tuple(
                OCR_BACKEND_LABELS[backend]
                for backend in self.ocr_backend_options
            ),
        )
        backend_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))
        backend_combo.bind("<<ComboboxSelected>>", self._change_ocr_backend)

        ttk.Label(ocr_controls, text="Keystroke delay (ms):").grid(
            row=1, column=2, sticky="w", padx=(18, 5), pady=(6, 0)
        )
        keystroke_spinbox = ttk.Spinbox(
            ocr_controls,
            from_=0,
            to=1000,
            increment=1,
            width=7,
            textvariable=self.keystroke_delay_var,
            command=self._apply_ocr_settings,
        )
        keystroke_spinbox.grid(row=1, column=3, sticky="w", pady=(6, 0))
        keystroke_spinbox.bind("<Return>", self._apply_ocr_settings)
        keystroke_spinbox.bind("<FocusOut>", self._apply_ocr_settings)

        filter_controls = ttk.LabelFrame(self, text="Green word filter", padding=8)
        filter_controls.pack(fill="x", padx=12, pady=(0, 8))
        filter_controls.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            filter_controls,
            text="Show only green text (white on black)",
            variable=self.filter_enabled_var,
            command=self._apply_filter,
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(
            filter_controls,
            text="Clean text rows",
            variable=self.keep_text_bands_var,
            command=self._apply_filter,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 6))

        ttk.Label(filter_controls, text="Hue range:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        hue_row = ttk.Frame(filter_controls)
        hue_row.grid(row=2, column=1, sticky="w", pady=2)
        self._make_filter_spinbox(hue_row, self.hue_min_var, 0, 359).pack(
            side="left"
        )
        ttk.Label(hue_row, text="to").pack(side="left", padx=5)
        self._make_filter_spinbox(hue_row, self.hue_max_var, 0, 359).pack(
            side="left"
        )

        ttk.Label(filter_controls, text="Minimum saturation:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        saturation_row = ttk.Frame(filter_controls)
        saturation_row.grid(row=3, column=1, sticky="w", pady=2)
        self._make_filter_spinbox(
            saturation_row, self.saturation_min_var, 0, 100
        ).pack(side="left")
        ttk.Label(saturation_row, text="%").pack(side="left", padx=(4, 0))

        ttk.Label(filter_controls, text="Minimum brightness:").grid(
            row=4, column=0, sticky="w", pady=2
        )
        brightness_row = ttk.Frame(filter_controls)
        brightness_row.grid(row=4, column=1, sticky="w", pady=2)
        self._make_filter_spinbox(
            brightness_row, self.value_min_var, 0, 100
        ).pack(side="left")
        ttk.Label(brightness_row, text="%").pack(side="left", padx=(4, 0))

        ttk.Label(filter_controls, text="White-mask morphology:").grid(
            row=5, column=0, sticky="w", pady=2
        )
        morphology_row = ttk.Frame(filter_controls)
        morphology_row.grid(row=5, column=1, sticky="w", pady=2)
        ttk.Label(morphology_row, text="Erode:").pack(side="left")
        self._make_filter_spinbox(
            morphology_row, self.erosion_iterations_var, 0, 5
        ).pack(side="left", padx=(4, 12))
        ttk.Label(morphology_row, text="Dilate:").pack(side="left")
        self._make_filter_spinbox(
            morphology_row, self.dilation_iterations_var, 0, 5
        ).pack(side="left", padx=(4, 5))
        ttk.Label(morphology_row, text="iterations (0 = off)").pack(side="left")

        ttk.Label(filter_controls, text="Minimum grouped blob area:").grid(
            row=6, column=0, sticky="w", pady=2
        )
        blob_area_row = ttk.Frame(filter_controls)
        blob_area_row.grid(row=6, column=1, sticky="w", pady=2)
        self._make_filter_spinbox(
            blob_area_row, self.minimum_blob_area_var, 0, 100000, width=7
        ).pack(side="left")
        ttk.Label(blob_area_row, text="white pixels (processed resolution)").pack(
            side="left", padx=(5, 0)
        )

        info = ttk.Frame(self, padding=(12, 0, 12, 10))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.region_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.status_var).pack(anchor="w", pady=(3, 0))

        self.preview = ttk.Label(self, anchor="center", text="Live preview")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _make_filter_spinbox(
        self,
        parent: ttk.Frame,
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
        width: int = 4,
    ) -> ttk.Spinbox:
        spinbox = ttk.Spinbox(
            parent,
            from_=minimum,
            to=maximum,
            width=width,
            textvariable=variable,
            command=self._apply_filter,
        )
        spinbox.bind("<Return>", self._apply_filter)
        spinbox.bind("<FocusOut>", self._apply_filter)
        return spinbox
