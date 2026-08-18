from typing import Any
import queue


def put_latest(target, value: Any) -> str:
    try:
        target.put_nowait(value)
        return "queued"
    except queue.Full:
        pass
    replaced = False
    try:
        target.get_nowait()
        replaced = True
    except queue.Empty:
        pass
    try:
        target.put_nowait(value)
        return "replaced" if replaced else "queued"
    except queue.Full:
        return "dropped"
