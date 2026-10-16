"""SeedVR2 upscaling of a still, by overlapping tiles, fully on the GPU.

SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0) is a video model; a photo is upscaled
by cutting it into equal square tiles, writing them as a short lossless video (FFV1, through
OpenCV), running the engine's own CLI over it with batch size 1 (every tile independent, models
loaded once), and feather-blending the upscaled tiles back together.

The CLI runs as a child process. A guard watches the GPU: if VRAM stays pinned at the limit while
the card draws little power, the Windows driver is paging into shared system memory (tens of
times slower); the job is killed and reported instead of crawling for hours.
"""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .images import ramp, read, tile_positions

VAE_ARGS = ["--vae_encode_tiled", "--vae_encode_tile_size", "1024", "--vae_decode_tiled",
            "--vae_decode_tile_size", "512", "--vae_decode_tile_overlap", "64", "--attention_mode", "sdpa"]
MAX_TILE_OUT = 2048   # output tile side measured to fit in 16 GB with every upscaler


class SeedVR2Error(RuntimeError):
    pass


# --- GPU watch -----------------------------------------------------------------------------------

def _gpu_index() -> str:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v.isdigit() else "0"


def gpu_state():
    """(used MB, total MB, power W or None, power limit W or None) of the GPU the engine uses."""
    out = subprocess.run(["nvidia-smi", "-i", _gpu_index(), "--query-gpu=memory.used,memory.total,power.draw,power.limit",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None
    return int(float(f[0])), int(float(f[1])), num(f[2]), num(f[3])


class SpillGuard(threading.Thread):
    """Tracks peak VRAM; kills the job after `grace` s of full VRAM at low power (a spill).
    "Low power" is under 35% of the card's power limit. If the card doesn't report power, the
    guard only tracks peak VRAM and says so."""

    def __init__(self, proc, grace=20.0, interval=0.25):
        super().__init__(daemon=True)
        self.proc, self.grace, self.interval = proc, grace, interval
        self.peak, self.spilled, self.stop = 0, False, threading.Event()
        self.errors, self.no_power = 0, False

    def run(self):
        since = None
        while not self.stop.is_set():
            try:
                used, total, power, limit = gpu_state()
                self.peak = max(self.peak, used)
                if power is None:
                    self.no_power = True
                    since = None
                elif used > total - 500 and power < 0.35 * (limit or 360):
                    since = since or time.monotonic()
                    if time.monotonic() - since > self.grace and self.proc.poll() is None:
                        self.spilled = True
                        _kill(self.proc)
                else:
                    since = None
            except Exception:
                self.errors += 1
            time.sleep(self.interval)


def _kill(proc):
    if proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


# --- paths -----------------------------------------------------------------------------------------

def _short_path(p: Path) -> str:
    try:
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024):
            return buf.value
    except (AttributeError, OSError):
        pass
    return str(p)


def ascii_workdir() -> Path:
    """A fresh empty folder whose path is pure ASCII. The engine saves frames with OpenCV, which
    cannot write to non-ASCII paths on Windows, so a photo called "Crăciun.jpg" (or a repo under a
    non-ASCII user name) must not put its frames there."""
    for base in (config.CACHE, Path(tempfile.gettempdir())):
        base.mkdir(parents=True, exist_ok=True)
        for cand in (str(base), _short_path(base)):
            if cand.isascii():
                return Path(tempfile.mkdtemp(prefix="seedvr2_", dir=cand))  # unique: runs may overlap
    raise SeedVR2Error("No ASCII-only folder available for SeedVR2's frames: move the repo to a path "
                       "without accented characters, or set TEMP to one.")


def _check_install(dit: str):
    cli = config.ENGINE / "inference_cli.py"
    if not cli.exists():
        raise SeedVR2Error(f"SeedVR2 engine not found ({cli}). Run install.ps1.")
    if not (config.SEEDVR2_MODELS / dit).exists():
        raise SeedVR2Error(f"Upscaler weights {dit} not installed. Run: install.ps1 -Upscaler <name> "
                           f"(installed: {', '.join(config.installed_upscalers()) or 'none'}).")
    if not (config.SEEDVR2_MODELS / "ema_vae_fp16.safetensors").exists():
        raise SeedVR2Error("SeedVR2 VAE missing. Run install.ps1.")
    return cli


# --- running the engine ------------------------------------------------------------------------------

# The engine's lines worth showing as progress: its phases, each tile, and the end.
PROGRESS = re.compile(r"Phase \d+: [^━]*?(?=\s*━|$)|Upscaling batch \d+/\d+|completed successfully in [\d.]+s")


def progress_line(line: str) -> str | None:
    m = PROGRESS.search(line)
    return None if not m else "  SeedVR2: " + m.group(0).strip().replace("Upscaling batch", "upscaling tile")


def upscale_tiles(tiles: list[np.ndarray], out_px: int, log_path: Path, model: str | None = None,
                  progress=print) -> tuple[list[np.ndarray], dict]:
    """Upscale equal square tiles to out_px x out_px each with upscaler `model` ('7b', '7b-sharp',
    '3b'; default: the one chosen at install). The engine's log is written to log_path, and its
    phases and tiles are reported through progress()."""
    dit = config.upscaler_file(model)
    cli = _check_install(dit)
    work = ascii_workdir()
    try:
        mkv = work / "tiles.mkv"
        t = tiles[0].shape[0]
        vw = cv2.VideoWriter(str(mkv), cv2.VideoWriter_fourcc(*"FFV1"), 24, (t, t))
        if not vw.isOpened():
            raise SeedVR2Error("OpenCV could not open an FFV1 writer")
        for tile in tiles:
            vw.write(tile)
        vw.release()
        raw = work / "raw"
        cmd = [sys.executable, str(cli), str(mkv), "--output", str(raw), "--output_format", "png",
               "--resolution", str(out_px), "--model_dir", str(config.SEEDVR2_MODELS), "--dit_model", dit,
               "--batch_size", "1", "--temporal_overlap", "0", *VAE_ARGS]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(cmd, cwd=config.ENGINE, env=config.cache_env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            guard = SpillGuard(proc)
            guard.start()
            try:
                for line in proc.stdout:  # every line to the log; the progress ones to the window too
                    log.write(line)
                    log.flush()
                    shown = progress_line(line.decode("utf-8", "replace"))
                    if shown:
                        progress(shown)
                proc.wait()
            finally:  # Ctrl+C or any error here must not leave the engine holding the GPU
                _kill(proc)
                guard.stop.set()
                guard.join()
        stats = {"tiles": len(tiles), "tile_in": t, "tile_out": out_px, "model": dit,
                 "seconds": round(time.perf_counter() - t0, 1), "gpu_peak_mb": guard.peak}
        if guard.no_power or guard.errors:
            stats["guard"] = "power not reported; spill guard inactive" if guard.no_power else "GPU readings failed"
        if guard.spilled:
            raise SeedVR2Error(f"SeedVR2 ran out of VRAM and spilled into system memory; stopped. Log: {log_path}")
        frames = sorted(raw.rglob("*.png")) if raw.exists() else []
        if proc.returncode != 0 or len(frames) != len(tiles):
            raise SeedVR2Error(f"SeedVR2 failed (exit {proc.returncode}, {len(frames)}/{len(tiles)} tiles). "
                               f"Log: {log_path}")
        out = []
        for f in frames:
            im = read(f)
            out.append(im if im.shape[:2] == (out_px, out_px)
                       else cv2.resize(im, (out_px, out_px), interpolation=cv2.INTER_LANCZOS4))
        return out, stats
    finally:
        shutil.rmtree(work, ignore_errors=True)


def tile_plan(W: int, H: int, scale: int, tile: int = 1024, overlap: int = 128):
    """Square tile side, overlap, and the tile offsets along x and y. The tile is 1024 px at 2x and
    smaller at 3x/4x so the upscaled tile stays at MAX_TILE_OUT (measured to fit in 16 GB); a photo
    smaller than one tile uses its short side."""
    t = min(tile, MAX_TILE_OUT // scale, W, H)
    ov = min(overlap, t // 4)
    return t, ov, tile_positions(W, t, ov), tile_positions(H, t, ov)


def blend(H: int, W: int, scale: int, t: int, xs: list[int], ys: list[int], tiles) -> np.ndarray:
    """Feather-blend upscaled tiles (in row-major order of ys x xs) into one image. Each tile ramps
    linearly across the overlap with its actual neighbours and stays flat at the image border."""
    T = t * scale

    def axis_ramp(pos, i):
        r = np.ones(T, np.float32)
        if i > 0:
            ov = (pos[i - 1] + t - pos[i]) * scale
            r[:ov] = ramp(2 * ov, ov)[:ov]
        if i < len(pos) - 1:
            ov = (pos[i] + t - pos[i + 1]) * scale
            r[T - ov:] = np.minimum(r[T - ov:], ramp(2 * ov, ov)[ov:])
        return r

    canvas = np.zeros((H * scale, W * scale, 3), np.float32)
    weight = np.zeros((H * scale, W * scale, 1), np.float32)
    k = 0
    for j, y in enumerate(ys):
        wy = axis_ramp(ys, j)
        for i, x in enumerate(xs):
            wx = axis_ramp(xs, i)
            w2 = (wy[:, None] * wx[None, :])[:, :, None]
            Y, X = y * scale, x * scale
            canvas[Y:Y + T, X:X + T] += tiles[k].astype(np.float32) * w2
            weight[Y:Y + T, X:X + T] += w2
            k += 1
    return np.clip(canvas / np.maximum(weight, 1e-6) + 0.5, 0, 255).astype(np.uint8)


def upscale(img: np.ndarray, scale: int, log_path: Path, tile: int = 1024, overlap: int = 128,
            model: str | None = None, progress=print) -> tuple[np.ndarray, dict]:
    """Upscale a whole image by an integer factor with overlapping tiles and feathered seams.
    1024 px tiles at 2x give the model enough context to recognise faces (small tiles make soft,
    grainy results)."""
    H, W = img.shape[:2]
    t, ov, xs, ys = tile_plan(W, H, scale, tile, overlap)
    crops = [img[y:y + t, x:x + t] for y in ys for x in xs]
    tiles, stats = upscale_tiles(crops, t * scale, log_path, model, progress)
    stats.update(scale=scale, grid=[len(xs), len(ys)])
    return blend(H, W, scale, t, xs, ys, tiles), stats
