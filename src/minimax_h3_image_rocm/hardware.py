"""GPU-family profiles: architecture identity, device policy and foundation lock."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

DEFAULT_PROFILE = "r9700"


def load_profile(root: Path, name: str | None = None) -> dict:
    name = name or os.environ.get("H3I_HARDWARE") or DEFAULT_PROFILE
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"Invalid hardware profile name: {name!r}")
    path = Path(root) / "configs/hardware" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown hardware profile: {name}")
    profile = json.loads(path.read_text())
    if profile.get("name") != name:
        raise ValueError(f"Hardware profile name mismatch in {path}")
    return profile


def load_runtime_lock(root: Path, profile: dict) -> dict:
    return json.loads((Path(root) / profile["runtime_lock"]).read_text())
