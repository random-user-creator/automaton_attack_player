from __future__ import annotations

import multiprocessing as mp
import signal

from .platform.windows import enable_dpi_awareness
from .ui.application import DotaWordGameApp


def main() -> None:
    mp.freeze_support()
    enable_dpi_awareness()
    app = DotaWordGameApp()
    signal.signal(signal.SIGINT, lambda _signal, _frame: app.after(0, app._on_close))
    try:
        app.mainloop()
    finally:
        if not app._closing:
            app._on_close()
