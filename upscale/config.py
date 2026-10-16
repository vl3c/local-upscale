"""Paths and settings. Everything heavy lives inside the repo folder (and is git-ignored)."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"          # put your photos here (ignored by git)
OUTPUT = ROOT / "output"        # results, one folder per photo (ignored by git)
ENGINE = ROOT / "engine" / "SeedVR2"
MODELS = ROOT / "models"
SEEDVR2_MODELS = MODELS / "seedvr2"
FACE_MODEL = MODELS / "face" / "face_detection_yunet_2023mar.onnx"
CACHE = ROOT / ".cache"

MANIFEST = ROOT / "models.json"


def local_config_error() -> str | None:
    """Why config.local.json can't be read, or None if it is fine (or absent)."""
    p = ROOT / "config.local.json"
    try:
        if p.exists():
            json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        return f"config.local.json is not valid JSON ({e}); its settings are ignored. Write backslashes as \\\\."
    return None


def _local_config() -> dict:
    p = ROOT / "config.local.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))  # tolerate a BOM (Notepad, PowerShell 5.1)
        except json.JSONDecodeError:
            pass
    return {}


def local_image_dir() -> Path:
    """The local-image install (Qwen-Image through ComfyUI). Override with the LOCAL_IMAGE_DIR
    environment variable or "local_image_dir" in config.local.json."""
    v = os.environ.get("LOCAL_IMAGE_DIR") or _local_config().get("local_image_dir")
    return Path(v).expanduser() if v else Path.home() / ".claude" / "skills" / "local-image"


def cache_env() -> dict:
    """Environment for child processes: every cache inside the repo, UTF-8 console output."""
    return dict(os.environ,
                HF_HOME=str(CACHE / "huggingface"), HF_HUB_CACHE=str(CACHE / "huggingface" / "hub"),
                TORCH_HOME=str(CACHE / "torch"), TRITON_CACHE_DIR=str(CACHE / "triton"),
                PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def upscaler_names() -> list[str]:
    return list(manifest()["upscalers"])


def default_upscaler() -> str:
    """The upscaler chosen at install (config.local.json "upscaler"), else the manifest default."""
    m = manifest()
    choice = _local_config().get("upscaler") or m["default_upscaler"]
    return choice if choice in m["upscalers"] else m["default_upscaler"]


def upscaler_file(name: str | None = None) -> str:
    """Weights filename for an upscaler key ('7b', '7b-sharp', '3b')."""
    m = manifest()["upscalers"]
    key = name or default_upscaler()
    if key not in m:
        raise ValueError(f"unknown upscaler {key!r}; choose from {', '.join(m)}")
    return m[key]["name"]


def installed_upscalers() -> list[str]:
    m = manifest()["upscalers"]
    return [k for k, v in m.items() if (ROOT / v["dest"] / v["name"]).exists()]
