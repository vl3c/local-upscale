"""Image I/O (Unicode-safe on Windows), labels, and the size arithmetic the workflow relies on."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

QWEN_MAX = 2048      # local-image renders at most ~2048x2048 (~4.2 MP)
QWEN_MIN = 1024      # below this Qwen has too few pixels to restore faces
FINAL_MIN_LONG = 2048  # small originals are delivered at 2K (long side)


def read(path) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return im


def save(path, im) -> Path:
    path = Path(path)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, im, params)
    if not ok:
        raise OSError(f"cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return path


def resize(im, w, h) -> np.ndarray:
    """Area averaging when shrinking (keeps detail, no aliasing), Lanczos when enlarging."""
    if (im.shape[1], im.shape[0]) == (w, h):
        return im
    shrink = w < im.shape[1] or h < im.shape[0]
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)


def label(im, text) -> np.ndarray:
    im = im.copy()
    fs = max(im.shape[1] / 1600, 0.5) * 1.4
    org = (int(12 * fs), int(34 * fs))
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, round(4 * fs)), cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, round(1.5 * fs)), cv2.LINE_AA)
    return im


def to_height(im, h) -> np.ndarray:
    return resize(im, round(im.shape[1] * h / im.shape[0]), h)


def grid(images, cols) -> np.ndarray:
    """Tile equally sized images into rows of `cols`, padding with black."""
    tiles = list(images)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])


def unsharp(im, amount=0.35, sigma=0.8) -> np.ndarray:
    f = im.astype(np.float32)
    return np.clip(f + amount * (f - cv2.GaussianBlur(f, (0, 0), sigma)), 0, 255).astype(np.uint8)


# --- sizes -------------------------------------------------------------------------------------

def qwen_scale(w: int, h: int) -> int:
    """Qwen's edit "scale" (about sqrt of the output area): the photo's own size, within its limits."""
    return int(min(QWEN_MAX, max(QWEN_MIN, round(math.sqrt(w * h) / 16) * 16)))


def final_size(w: int, h: int) -> tuple[int, int]:
    """The original resolution, or 2K on the long side if the original is smaller."""
    if max(w, h) >= FINAL_MIN_LONG:
        return w, h
    k = FINAL_MIN_LONG / max(w, h)
    return round(w * k), round(h * k)


def sr_factor(target: tuple[int, int], source: tuple[int, int]) -> int:
    """Integer SeedVR2 factor so its output covers the target: at least 2x (supersampling), at most 4x.
    In practice 2x unless the original is very large (over ~4700 px wide); beyond 4x the rest is a
    plain resize."""
    return min(4, max(2, math.ceil(max(target[0] / source[0], target[1] / source[1]))))


def tile_positions(size: int, tile: int, min_overlap: int) -> list[int]:
    """Start offsets of equally spaced tiles covering [0, size), first flush with 0 and last flush
    with the end, every neighbouring pair overlapping by at least min_overlap."""
    if size <= tile:
        return [0]
    n = math.ceil((size - min_overlap) / (tile - min_overlap))
    return [round(i * (size - tile) / (n - 1)) for i in range(n)]


def ramp(n: int, overlap: int) -> np.ndarray:
    """1-D blend weight for a tile: linear ramps over the overlaps, flat in the middle."""
    w = np.ones(n, np.float32)
    if overlap > 0:
        r = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w
