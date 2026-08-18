from __future__ import annotations

import ctypes
import queue
import re
import signal
import time

from ..logging import timing_log
from ..queueing import put_latest

class WindowsScanCodeSender:
    """Fast native keyboard injection using one SendInput batch when possible."""

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008
    MAPVK_VK_TO_VSC = 0

    def __init__(self) -> None:
        if not hasattr(ctypes, "WinDLL"):
            raise RuntimeError("Native SendInput typing is only available on Windows.")

        from ctypes import wintypes

        ulong_ptr = wintypes.WPARAM

        class KeyboardInput(ctypes.Structure):
            _fields_ = (
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            )

        class MouseInput(ctypes.Structure):
            _fields_ = (
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            )

        class HardwareInput(ctypes.Structure):
            _fields_ = (
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            )

        class InputUnion(ctypes.Union):
            # INPUT's cbSize must include its largest native union member even
            # though this app only submits keyboard records.
            _fields_ = (
                ("mi", MouseInput),
                ("ki", KeyboardInput),
                ("hi", HardwareInput),
            )

        class Input(ctypes.Structure):
            _anonymous_ = ("data",)
            _fields_ = (("type", wintypes.DWORD), ("data", InputUnion))

        self._input_type = Input
        self._keyboard_input_type = KeyboardInput
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(Input),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = wintypes.UINT
        self._user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
        self._user32.MapVirtualKeyW.restype = wintypes.UINT

    def _events(self, letters: str):
        events = []
        for letter in letters:
            virtual_key = ord(letter.upper())
            scan_code = self._user32.MapVirtualKeyW(
                virtual_key, self.MAPVK_VK_TO_VSC
            )
            if not scan_code:
                raise OSError(f"No keyboard scan code for {letter!r}.")
            events.append(
                self._input_type(
                    type=self.INPUT_KEYBOARD,
                    ki=self._keyboard_input_type(
                        0, scan_code, self.KEYEVENTF_SCANCODE, 0, 0
                    ),
                )
            )
            events.append(
                self._input_type(
                    type=self.INPUT_KEYBOARD,
                    ki=self._keyboard_input_type(
                        0,
                        scan_code,
                        self.KEYEVENTF_SCANCODE | self.KEYEVENTF_KEYUP,
                        0,
                        0,
                    ),
                )
            )
        return events

    def _send_events(self, events) -> None:
        if not events:
            return
        array_type = self._input_type * len(events)
        event_array = array_type(*events)
        sent = self._user32.SendInput(
            len(event_array), event_array, ctypes.sizeof(self._input_type)
        )
        if sent != len(event_array):
            error = ctypes.get_last_error()
            raise ctypes.WinError(error or 1)

    def send(self, letters: str, delay_ms: float, stop_event) -> int:
        if not letters:
            return 0
        delay_seconds = max(0.0, delay_ms / 1000.0)
        if delay_seconds == 0.0:
            self._send_events(self._events(letters))
            return len(letters)

        for index, letter in enumerate(letters):
            if index:
                # Event.wait() has coarse timing in this Windows subprocess
                # and turned a requested 1ms gap into roughly 15ms. Modern
                # Python's Windows sleep uses a high-resolution waitable timer.
                time.sleep(delay_seconds)
                if stop_event.is_set():
                    return index
            self._send_events(self._events(letter))
        return len(letters)


def typing_process_main(stop_event, typing_commands, messages, errors) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        keyboard = WindowsScanCodeSender()
        timing_log("TYPE", "keyboard_ready", backend="windows_sendinput")
    except Exception as exc:
        put_latest(errors, f"Keyboard initialization failed: {exc}")
        return

    typed_recently: dict[str, float] = {}
    while not stop_event.is_set():
        try:
            (
                frame_id,
                captured_at,
                ocr_finished_at,
                words,
                target_window,
                keystroke_delay_ms,
            ) = typing_commands.get(timeout=0.1)
        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError, OSError):
            return

        dequeued_at = time.perf_counter()
        typing_queue_ms = (dequeued_at - ocr_finished_at) * 1000
        if not is_window_focused(target_window):
            timing_log(
                "TYPE",
                "skipped_not_focused",
                frame=frame_id,
                queue_ms=f"{typing_queue_ms:.1f}",
            )
            continue
        now = time.monotonic()
        typed_recently = {
            word: timestamp
            for word, timestamp in typed_recently.items()
            if now - timestamp < 3.0
        }
        typed: list[str] = []
        typing_started = time.perf_counter()
        character_count = 0
        try:
            pending_words: list[tuple[str, str]] = []
            for word in words:
                letters_only = re.sub(r"[^A-Za-z]", "", word)
                if not letters_only:
                    continue
                key = letters_only.casefold()
                if now - typed_recently.get(key, 0.0) < 1.5:
                    continue
                letters = letters_only.lower()
                pending_words.append((letters_only, letters))

            combined_letters = "".join(letters for _word, letters in pending_words)
            character_count = keyboard.send(
                combined_letters, keystroke_delay_ms, stop_event
            )
            emitted_count = 0
            for letters_only, letters in pending_words:
                emitted_count += len(letters)
                if emitted_count > character_count:
                    break
                key = letters_only.casefold()
                typed_recently[key] = now
                typed.append(letters_only)
        except Exception as exc:
            timing_log("TYPE", "output_failed", frame=frame_id, error=repr(exc))
            put_latest(errors, f"Keyboard output failed: {exc}")
            continue
        typing_finished = time.perf_counter()
        timing_log(
            "TYPE",
            "command_complete",
            frame=frame_id,
            queue_ms=f"{typing_queue_ms:.1f}",
            typing_ms=f"{(typing_finished - typing_started) * 1000:.1f}",
            ocr_to_type_ms=f"{(typing_finished - ocr_finished_at) * 1000:.1f}",
            capture_to_type_ms=f"{(typing_finished - captured_at) * 1000:.1f}",
            words=len(typed),
            chars=character_count,
            key_delay_ms=f"{keystroke_delay_ms:.1f}",
            backend="windows_sendinput",
        )
        if typed:
            put_latest(messages, f"Typed: {', '.join(typed)}")


def is_window_focused(handle: int | None) -> bool:
    if not handle or not hasattr(ctypes, "windll"):
        return False
    return int(ctypes.windll.user32.GetForegroundWindow()) == handle
