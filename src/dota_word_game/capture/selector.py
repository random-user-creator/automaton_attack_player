from __future__ import annotations

import ctypes
import tkinter as tk

from ..platform.windows import window_at_point
from .models import CaptureRegion

def virtual_screen_bounds(root: tk.Tk) -> tuple[int, int, int, int]:
    """Return the virtual desktop bounds, including negative monitor offsets."""
    if hasattr(ctypes, "windll"):
        user32 = ctypes.windll.user32
        return (
            user32.GetSystemMetrics(76),  # SM_XVIRTUALSCREEN
            user32.GetSystemMetrics(77),  # SM_YVIRTUALSCREEN
            user32.GetSystemMetrics(78),  # SM_CXVIRTUALSCREEN
            user32.GetSystemMetrics(79),  # SM_CYVIRTUALSCREEN
        )
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


class RegionSelector(tk.Toplevel):
    def __init__(self, parent: tk.Tk, on_selected) -> None:
        super().__init__(parent)
        self.on_selected = on_selected
        self.start_x = 0
        self.start_y = 0
        self.rectangle: int | None = None

        left, top, width, height = virtual_screen_bounds(parent)
        self.virtual_left = left
        self.virtual_top = top

        self.overrideredirect(True)
        self.geometry(f"{width}x{height}{left:+d}{top:+d}")
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.35)
        self.configure(cursor="crosshair")

        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            width // 2,
            36,
            text="Click and drag to select a capture area  •  Esc to cancel",
            fill="white",
            font=("Segoe UI", 16, "bold"),
        )

        self.canvas.bind("<ButtonPress-1>", self._start_selection)
        self.canvas.bind("<B1-Motion>", self._update_selection)
        self.canvas.bind("<ButtonRelease-1>", self._finish_selection)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.focus_force()
        self.grab_set()

    def _start_selection(self, event: tk.Event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        self.rectangle = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#39ff88",
            width=3,
        )

    def _update_selection(self, event: tk.Event) -> None:
        if self.rectangle is not None:
            self.canvas.coords(
                self.rectangle, self.start_x, self.start_y, event.x, event.y
            )

    def _finish_selection(self, event: tk.Event) -> None:
        x1, x2 = sorted((self.start_x, event.x))
        y1, y2 = sorted((self.start_y, event.y))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        region = CaptureRegion(
            left=self.virtual_left + x1,
            top=self.virtual_top + y1,
            width=x2 - x1,
            height=y2 - y1,
        )
        self.grab_release()
        self.withdraw()
        self.update_idletasks()
        target_window = window_at_point(
            region.left + region.width // 2,
            region.top + region.height // 2,
        )
        self.destroy()
        self.on_selected(region, target_window)
