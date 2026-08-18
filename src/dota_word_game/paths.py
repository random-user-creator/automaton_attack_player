from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

if FROZEN:
    local_app_data = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    STATE_PATH = local_app_data / "AutomatonAttackPlayer" / "app_state.json"
else:
    STATE_PATH = PROJECT_ROOT / "app_state.json"


def bundled_model_dir(model_name: str) -> Path | None:
    """Return an embedded model directory in packaged builds, if present."""
    candidate = RESOURCE_ROOT / "models" / model_name
    return candidate if candidate.is_dir() else None
