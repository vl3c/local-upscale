"""The local-image tool (Qwen-Image through ComfyUI), driven through its own CLI.

local-image is a separate install (it brings its own ComfyUI, models and server). This module only
calls its scripts/generate.py and stop.cmd; see config.local_image_dir() for where it is looked up.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config


class LocalImageMissing(RuntimeError):
    pass


class QwenError(RuntimeError):
    pass


def paths() -> tuple[Path, Path]:
    d = config.local_image_dir()
    gen, stop_cmd = d / "scripts" / "generate.py", d / "stop.cmd"
    if not gen.exists():
        raise LocalImageMissing(
            f"local-image not found at {d} (no scripts/generate.py). Install it, then point "
            "LOCAL_IMAGE_DIR (or local_image_dir in config.local.json) at its folder."
            + (f" Note: {config.local_config_error()}" if config.local_config_error() else ""))
    return gen, stop_cmd


def gpu_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def edit(prompt: str, images: list[Path], out_dir: Path, scale: int, count: int, seed: int,
         timeout: int = 3600) -> list[Path]:
    """One instruction edit: images[0] is the photo, any further images are references.
    Returns the new PNGs (RGB, metadata stripped) written to out_dir."""
    gen, _ = paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("*.png"))
    cmd = [sys.executable, str(gen), prompt]
    for im in images:
        cmd += ["--edit", str(Path(im).resolve())]
    cmd += ["--scale", str(scale), "--count", str(count), "--seed", str(seed), "--rgb", "--out", str(out_dir),
            "--timeout", str(timeout)]
    r = subprocess.run(cmd)
    made = sorted(set(out_dir.glob("*.png")) - before)
    if r.returncode != 0 or len(made) != count:
        raise QwenError(f"local-image edit failed (exit {r.returncode}, {len(made)}/{count} images); "
                        "see its server window or logs/server.log in the local-image folder")
    return made


def stop(detached: bool = False) -> None:
    """Stop the local-image server (it holds ~15.5 GB of VRAM; SeedVR2 needs the card next).
    detached: start its stop.cmd and return at once, outside this console (for a closing window)."""
    _, stop_cmd = paths()
    if not stop_cmd.exists():
        return
    if detached:
        from .window import detached as start_detached
        start_detached(["cmd", "/c", str(stop_cmd)])
    else:
        subprocess.run(["cmd", "/c", str(stop_cmd)], capture_output=True)
