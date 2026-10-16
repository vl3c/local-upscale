"""Faces: detection (OpenCV YuNet), close-up strips for choosing variants, and head mixing."""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .images import label, resize

Box = tuple[int, int, int, int]  # x, y, w, h


def detect(img, max_faces: int = 12, threshold: float = 0.6) -> list[Box]:
    """Faces in `img`, most confident first, overlaps removed, returned left to right.
    Empty if the detector model is not installed or nothing is found."""
    if not config.FACE_MODEL.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return []
    H, W = img.shape[:2]
    k = min(1.0, 1280 / max(H, W))
    small = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN.create(str(config.FACE_MODEL), "", (small.shape[1], small.shape[0]), threshold)
    _, found = det.detect(small)
    picked: list[Box] = []
    for r in sorted(found if found is not None else [], key=lambda r: -r[14]):
        x, y, w, h = (int(v / k) for v in r[:4])
        if w < W * 0.02:
            continue
        if all(abs(x - px) > max(w, pw) * 0.5 or abs(y - py) > max(h, ph) * 0.5 for px, py, pw, ph in picked):
            picked.append((x, y, w, h))
        if len(picked) == max_faces:
            break
    return sorted(picked)


def head_box(face: Box, W: int, H: int, scale: float = 2.4) -> tuple[int, int, int]:
    """A square (x0, y0, side) around a face with room above for hair, clipped to the image."""
    x, y, w, h = face
    side = min(int(max(w, h) * scale), W, H)
    cx, cy = x + w // 2, y + h // 2 - int(h * 0.15)
    x0 = int(np.clip(cx - side // 2, 0, W - side))
    y0 = int(np.clip(cy - side // 2, 0, H - side))
    return x0, y0, side


def color_match(src, ref) -> np.ndarray:
    """Match src's LAB mean and spread to ref's (same-size regions)."""
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        s[..., c] = (s[..., c] - s[..., c].mean()) / (s[..., c].std() + 1e-6) * r[..., c].std() + r[..., c].mean()
    return cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def head_mask(face: Box, x0: int, y0: int, side: int) -> np.ndarray:
    """Feathered mask of the head (face and hair) inside the square box: zero below the chin, so
    collars and clothes always stay from the base image, and zero along every edge of the box, so
    the pasted head never ends in a hard line."""
    x, y, w, h = face
    m = np.zeros((side, side), np.float32)
    cv2.ellipse(m, (int(x - x0 + w / 2), int(y - y0 + h * 0.38)), (int(w * 0.95), int(h * 1.05)), 0, 0, 360, 1, -1)
    m[max(0, int(y - y0 + h * 1.02)):, :] = 0
    mg = max(2, int(side * 0.06))
    m[:mg], m[-mg:], m[:, :mg], m[:, -mg:] = 0, 0, 0, 0
    # The blur radius (mg - 1) is smaller than the zeroed margin, so the box edges stay exactly 0.
    return cv2.GaussianBlur(m, (2 * mg - 1, 2 * mg - 1), mg / 3)


def mix_heads(base, donors: dict[int, np.ndarray], faces: list[Box]) -> tuple[np.ndarray, list[str]]:
    """Replace head i (1-based, left to right in `faces`) of `base` with the same head from donors[i].
    Donors are restorations of the same photo, so heads line up to within a pixel or two; the small
    shift is measured (phase correlation) and corrected, then the head is colour-matched and blended."""
    H, W = base.shape[:2]
    out = base.astype(np.float32)
    notes = []
    for i, donor in donors.items():
        face = faces[i - 1]
        x0, y0, side = head_box(face, W, H)
        gray = lambda im: cv2.GaussianBlur(cv2.cvtColor(im[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2GRAY)
                                           .astype(np.float32), (0, 0), 2)
        window = cv2.createHanningWindow((side, side), cv2.CV_32F)  # unwindowed, shifts read ~2x too small
        (dx, dy), resp = cv2.phaseCorrelate(gray(base), gray(donor), window)
        if resp < 0.05 or max(abs(dx), abs(dy)) > 8:  # no reliable match: don't move the head
            dx = dy = 0.0
        donor = cv2.warpAffine(donor, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        region = out[y0:y0 + side, x0:x0 + side]
        new = color_match(donor[y0:y0 + side, x0:x0 + side], region.astype(np.uint8)).astype(np.float32)
        m = head_mask(face, x0, y0, side)[..., None]
        out[y0:y0 + side, x0:x0 + side] = region * (1 - m) + new * m
        notes.append(f"face {i}: aligned by ({dx:.1f}, {dy:.1f}) px")
    return np.clip(out, 0, 255).astype(np.uint8), notes


def square_crop(img, cx: int, cy: int, half: int) -> np.ndarray:
    """A 2*half square centred on (cx, cy); parts outside the image are black, never stretched."""
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    return pad[cy:cy + 2 * half, cx:cx + 2 * half]


def strip(images, labels, faces: list[Box] | None = None, max_rows: int = 3, side: int = 420):
    """One row per face, each showing the same square face crop in every image. Without `faces`,
    the largest `max_rows` faces of the first image are used, shown left to right.
    None when there are no faces."""
    if faces is None:
        found = detect(images[0])
        faces = sorted(sorted(found, key=lambda f: -f[2] * f[3])[:max_rows])
    if not faces:
        return None
    rows = []
    for x, y, w, h in faces[:max_rows]:
        cx, cy, half = x + w // 2, y + h // 2, max(8, int(max(w, h) * 0.9))
        rows.append(np.hstack([label(resize(square_crop(im, cx, cy, half), side, side), t)
                               for im, t in zip(images, labels)]))
    return np.vstack(rows)


def numbered(img, faces: list[Box]) -> np.ndarray:
    """The image with each detected face boxed and numbered (the numbers `mix --face` uses)."""
    out = img.copy()
    t = max(2, img.shape[1] // 500)
    for n, (x, y, w, h) in enumerate(faces, 1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), t)
        cv2.putText(out, str(n), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.8, (0, 0, 0), t * 3, cv2.LINE_AA)
        cv2.putText(out, str(n), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, t * 0.8, (0, 255, 255), t, cv2.LINE_AA)
    return out
