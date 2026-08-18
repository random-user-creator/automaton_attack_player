from __future__ import annotations

import ctypes

def enable_dpi_awareness() -> None:
    """Keep Tk coordinates aligned with physical screenshot pixels on Windows."""
    if not hasattr(ctypes, "windll"):
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def window_at_point(x: int, y: int) -> int | None:
    if not hasattr(ctypes, "windll"):
        return None
    handle = ctypes.windll.user32.WindowFromPoint(POINT(x, y))
    if not handle:
        return None
    return int(ctypes.windll.user32.GetAncestor(handle, 2) or handle)


def focus_window(handle: int | None) -> bool:
    if not handle or not hasattr(ctypes, "windll"):
        return False
    ctypes.windll.user32.ShowWindow(handle, 9)  # SW_RESTORE
    return bool(ctypes.windll.user32.SetForegroundWindow(handle))


def is_window_focused(handle: int | None) -> bool:
    if not handle or not hasattr(ctypes, "windll"):
        return False
    return int(ctypes.windll.user32.GetForegroundWindow()) == handle


def is_window_valid(handle: int | None) -> bool:
    if not handle or not hasattr(ctypes, "windll"):
        return False
    return bool(ctypes.windll.user32.IsWindow(handle))
