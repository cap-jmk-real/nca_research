"""Refuse nca_research artifact writes that land on the internal disk.

A path under ``/Volumes/...`` is not enough: if the Extreme SSD is unmounted,
``mkdir`` can create that folder on the boot volume. Compare device ids with
``/`` and ``/System/Volumes/Data`` before creating anything.
"""

from __future__ import annotations

import os
from pathlib import Path

VOLUME = Path("/Volumes/Extreme SSD")
DATA_ROOT = VOLUME / "nca_research"
RESULTS = DATA_ROOT / "results"


def _internal_devs() -> set[int]:
    devs = {os.stat("/").st_dev}
    data = Path("/System/Volumes/Data")
    if data.exists():
        devs.add(data.stat().st_dev)
    return devs


def _anchor(path: Path) -> Path:
    cursor = path
    while not cursor.exists():
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return cursor


def assert_external(path: Path) -> None:
    """Raise SystemExit unless ``path`` is on a mounted external volume."""
    resolved = path.resolve()
    text = str(resolved)
    if not text.startswith("/Volumes/"):
        raise SystemExit(f"refusing to write outside /Volumes: {text}")
    if "Macintosh HD" in text:
        raise SystemExit(f"refusing internal volume: {text}")
    anchor = _anchor(resolved)
    if not anchor.exists():
        raise SystemExit(f"refusing unmounted path: {text}")
    if anchor.stat().st_dev in _internal_devs():
        raise SystemExit(
            f"refusing path on the internal disk (volume is not an external mount): {text}"
        )


def cache_map(root: Path) -> dict[str, Path]:
    assert_external(root)
    hf = root / "hf-home"
    return {
        "HF_HOME": hf,
        "HF_HUB_CACHE": hf / "hub",
        "HUGGINGFACE_HUB_CACHE": hf / "hub",
        "HF_DATASETS_CACHE": hf / "datasets",
        "HF_XET_CACHE": hf / "xet",
        "TMPDIR": root / "tmp",
        "TORCH_HOME": root / "torch-home",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor-cache",
        "XDG_CACHE_HOME": root / "xdg-cache",
        "PYTHONPYCACHEPREFIX": root / "pycache",
        "PIP_CACHE_DIR": root / "pip-cache",
    }


def apply_cache_env(root: Path) -> dict[str, str]:
    """Point HF, torch, tmp, and pip caches at ``root`` and return the mapping."""
    mapping = cache_map(root)
    out: dict[str, str] = {}
    for key, dest in mapping.items():
        dest.mkdir(parents=True, exist_ok=True)
        value = str(dest)
        os.environ[key] = value
        out[key] = value
    return out
