from __future__ import annotations

import os
import threading
import time
from typing import Any


VERBOSE_TIMING = os.environ.get("DOTA_WORD_VERBOSE", "1").lower() not in {
    "0",
    "false",
    "off",
}


def timing_log(component: str, event: str, **values: Any) -> None:
    if not VERBOSE_TIMING:
        return
    timestamp = time.strftime("%H:%M:%S")
    milliseconds = int((time.time() % 1) * 1000)
    details = " ".join(f"{key}={value}" for key, value in values.items())
    print(
        f"{timestamp}.{milliseconds:03d} [{component}] "
        f"pid={os.getpid()} thread={threading.current_thread().name} "
        f"event={event} {details}".rstrip(),
        flush=True,
    )
