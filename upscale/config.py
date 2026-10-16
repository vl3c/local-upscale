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

SEEDVR2_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"


def _local_config() -> dict:
    p = ROOT / "config.local.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
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
