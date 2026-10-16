"""SeedVR2-7B upscaling of a still, by overlapping tiles, fully on the GPU.

SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0) is a video model; a photo is upscaled
by cutting it into equal square tiles, writing them as a short lossless video (FFV1, through
OpenCV), running the engine's own CLI over it with batch size 1 (every tile independent, models
loaded once), and feather-blending the upscaled tiles back together.

The CLI runs as a child process. A guard watches the GPU: if VRAM stays pinned at the limit while
the card draws little power, the Windows driver is paging into shared system memory (tens of
times slower); the job is killed and reported instead of crawling for hours.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .images import ramp, tile_positions

VAE_ARGS = ["--vae_encode_tiled", "--vae_encode_tile_size", "1024", "--vae_decode_tiled",
            "--vae_decode_tile_size", "512", "--vae_decode_tile_overlap", "64", "--attention_mode", "sdpa"]


class SeedVR2Error(RuntimeError):
    pass


def _gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    used, total, power = (float(x) for x in out.split(","))
    return int(used), int(total), power


class SpillGuard(threading.Thread):
    """Tracks peak VRAM; kills the job after `grace` s of full VRAM at low power (a spill)."""

    def __init__(self, proc, grace=20.0, interval=0.25):
        super().__init__(daemon=True)
        self.proc, self.grace, self.interval = proc, grace, interval
        self.peak, self.spilled, self.stop = 0, False, threading.Event()

    def run(self):
        since = None
        while not self.stop.is_set():
            try:
                used, total, power = _gpu_state()
                self.peak = max(self.peak, used)
                if used > total - 500 and power < 130:
                    since = since or time.monotonic()
                    if time.monotonic() - since > self.grace and self.proc.poll() is None:
                        self.spilled = True
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
                else:
                    since = None
            except Exception:
                pass
            time.sleep(self.interval)


def _check_install():
    cli = config.ENGINE / "inference_cli.py"
    if not cli.exists():
        raise SeedVR2Error(f"SeedVR2 engine not found ({cli}). Run install.ps1.")
    if not (config.SEEDVR2_MODELS / config.SEEDVR2_DIT).exists():
        raise SeedVR2Error(f"SeedVR2 weights missing in {config.SEEDVR2_MODELS}. Run install.ps1.")
    return cli


def upscale_tiles(tiles: list[np.ndarray], out_px: int, work: Path) -> tuple[list[np.ndarray], dict]:
    """Upscale equal square tiles to out_px x out_px each. Returns the tiles and run stats."""
    cli = _check_install()
    work.mkdir(parents=True, exist_ok=True)
    mkv = work / "tiles.mkv"
    t = tiles[0].shape[0]
    vw = cv2.VideoWriter(str(mkv), cv2.VideoWriter_fourcc(*"FFV1"), 24, (t, t))
    if not vw.isOpened():
        raise SeedVR2Error("OpenCV could not open an FFV1 writer")
    for tile in tiles:
        vw.write(tile)
    vw.release()
    raw = work / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    cmd = [sys.executable, str(cli), str(mkv), "--output", str(raw), "--output_format", "png",
           "--resolution", str(out_px), "--model_dir", str(config.SEEDVR2_MODELS), "--dit_model", config.SEEDVR2_DIT,
           "--batch_size", "1", "--temporal_overlap", "0", *VAE_ARGS]
    log_path = work / "seedvr2.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=config.ENGINE, env=config.cache_env(), stdout=log, stderr=subprocess.STDOUT)
        guard = SpillGuard(proc)
        guard.start()
        proc.wait()
        guard.stop.set()
        guard.join()
    stats = {"tiles": len(tiles), "tile_in": t, "tile_out": out_px, "seconds": round(time.perf_counter() - t0, 1),
             "gpu_peak_mb": guard.peak}
    if guard.spilled:
        raise SeedVR2Error(f"SeedVR2 ran out of VRAM and spilled into system memory; stopped. Log: {log_path}")
    frames = sorted(raw.rglob("*.png")) if raw.exists() else []
    if proc.returncode != 0 or len(frames) != len(tiles):
        raise SeedVR2Error(f"SeedVR2 failed (exit {proc.returncode}, {len(frames)}/{len(tiles)} tiles). Log: {log_path}")
    out = []
    for f in frames:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        out.append(im if im.shape[:2] == (out_px, out_px) else cv2.resize(im, (out_px, out_px), interpolation=cv2.INTER_LANCZOS4))
    return out, stats


def upscale(img: np.ndarray, scale: int, work: Path, tile: int = 1024, overlap: int = 128) -> tuple[np.ndarray, dict]:
    """Upscale a whole image by an integer factor with overlapping tiles and feathered seams.

    1024 px tiles give the model enough context to recognise faces (small tiles make soft,
    grainy results); a photo smaller than one tile is processed as a single square tile."""
    H, W = img.shape[:2]
    t = min(tile, W, H)
    ov = min(overlap, t // 4)
    step = t - ov
    xs, ys = tile_positions(W, t, step), tile_positions(H, t, step)
    boxes = [(x, y) for y in ys for x in xs]
    tiles, stats = upscale_tiles([img[y:y + t, x:x + t] for x, y in boxes], t * scale, work)
    T, OV = t * scale, ov * scale
    canvas = np.zeros((H * scale, W * scale, 3), np.float32)
    weight = np.zeros((H * scale, W * scale, 1), np.float32)
    for (x, y), up in zip(boxes, tiles):
        wy, wx = ramp(T, OV), ramp(T, OV)
        if y == 0:
            wy[:OV] = 1
        if y + t >= H:
            wy[-OV:] = 1
        if x == 0:
            wx[:OV] = 1
        if x + t >= W:
            wx[-OV:] = 1
        w2 = (wy[:, None] * wx[None, :])[:, :, None]
        Y, X = y * scale, x * scale
        canvas[Y:Y + T, X:X + T] += up.astype(np.float32) * w2
        weight[Y:Y + T, X:X + T] += w2
    out = np.clip(canvas / np.maximum(weight, 1e-6) + 0.5, 0, 255).astype(np.uint8)
    stats.update(scale=scale, grid=[len(xs), len(ys)])
    (work / "stats.json").write_text(json.dumps(stats, indent=2))
    return out, stats
