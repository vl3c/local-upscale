"""The photo workflow: variants -> (pick, optionally mix faces) -> finish.

Layout of output/<photo>/ (the top level is the set to flip through, in viewing order):

    <photo>_0_original_WxH.<ext>          the untouched original
    <photo>_1_original_<final>.png        the original resized to the final size (if it differs)
    <photo>_2_final_vN_<final>.png        THE RESULT: original resolution, or 2K if the original is smaller
    <photo>_3_final_vN_<final>_sharpened.png
    <photo>_4_upscaled_vN_<big>.png       the full SeedVR2 output (about 2x the Qwen output)
    variants/  variant_0_original_*, variant_N_WxH.png, contact_sheet.jpg, faces.jpg (+ faces_numbered.jpg from mix)
    work/      Qwen's native outputs, variants.json (seeds), SeedVR2 logs, comparisons
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np

from . import config, faces, prompts, qwen, seedvr2
from .images import (final_size, grid, label, qwen_scale, read, resize, save, sr_factor, to_height,
                     unsharp)

SHEET_W = 800


def folders(photo: Path) -> tuple[Path, Path, Path]:
    d = config.OUTPUT / photo.stem
    v, w = d / "variants", d / "work"
    for p in (d, v, w):
        p.mkdir(parents=True, exist_ok=True)
    return d, v, w


def native_path(work: Path, n) -> Path:
    return work / f"variant_{n}_native.png"


def existing_variants(work: Path) -> list[int]:
    out = []
    for p in work.glob("variant_*_native.png"):
        n = p.stem.split("_")[1]
        if n.isdigit():
            out.append(int(n))
    return sorted(out)


def variants(photo: Path, refs: list[Path] = (), colorize: bool = False, seeds: list[int] | None = None,
             keep_server: bool = False, log=print) -> Path:
    """Four Qwen restorations (two seeds x 2), delivered at the photo's own size for flipping."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    orig = read(photo)
    h, wd = orig.shape[:2]
    ext = photo.suffix.lower()
    shutil.copy(photo, d / f"{photo.stem}_0_original_{wd}x{h}{ext}")
    shutil.copy(photo, v / f"variant_0_original_{wd}x{h}{ext}")
    scale = qwen_scale(wd, h)
    used = qwen.gpu_used_mb()
    if used is not None and used > 4000:
        log(f"warning: {used} MB of VRAM already in use; local-image needs ~15.5 GB and may refuse to start")
    start = max(existing_variants(w), default=0)
    seeds = seeds or [random.randrange(1, 2**31) for _ in range(2)]
    prompt = prompts.pick(colorize, bool(refs))
    tmp = w / "_qwen"
    made = []
    try:
        for seed in seeds:  # 2 batches of 2: one batch of 4 at 2048 px overflows 16 GB and crawls
            log(f"Qwen edit: seed {seed}, scale {scale}")
            made += [(seed, p) for p in qwen.edit(prompt, [photo, *refs], tmp, scale, 2, seed)]
    finally:
        if not keep_server:
            qwen.stop()
    info_path = w / "variants.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"photo": photo.name, "variants": []}
    info.update(original_size=[wd, h], qwen_scale=scale)
    fulls = []
    for i, (seed, p) in enumerate(made, start + 1):
        native = read(p)
        shutil.move(str(p), str(native_path(w, i)))
        full = resize(native, wd, h)  # back to the exact original proportions (Qwen rounds to 16 px)
        save(v / f"variant_{i}_{wd}x{h}.png", full)
        fulls.append((i, full))
        info["variants"].append({"n": i, "seed": seed, "colorize": colorize, "refs": [r.name for r in refs],
                                 "native_size": [native.shape[1], native.shape[0]]})
    shutil.rmtree(tmp, ignore_errors=True)
    info_path.write_text(json.dumps(info, indent=2))
    write_overviews(photo)
    log(f"done: {v}")
    return v


def write_overviews(photo: Path) -> None:
    """contact_sheet.jpg and faces.jpg for the variants that still exist (the user may delete
    rejects from variants/). faces_numbered.jpg is written by `mix`, for the base it mixes into."""
    d, v, w = folders(photo)
    orig = read(photo)
    h, wd = orig.shape[:2]
    have = [(n, v / f"variant_{n}_{wd}x{h}.png") for n in existing_variants(w)]
    have = [(n, p) for n, p in have if p.exists()]
    if not have:
        return
    ims = [orig] + [read(p) for _, p in have]
    labels = ["original"] + [f"variant {n}" for n, _ in have]
    th = round(SHEET_W * h / wd)
    save(v / "contact_sheet.jpg", grid([label(resize(im, SHEET_W, th), t) for im, t in zip(ims, labels)], 3))
    found = faces.detect(ims[1], max_faces=6)
    s = faces.strip(ims, labels, faces=found[:3] if found else None)
    if s is not None:
        save(v / "faces.jpg", s)


def mix(photo: Path, base: int, picks: dict[int, int] | None, log=print) -> int | None:
    """New variant = `base` with face i taken from variant picks[i]. Without picks, only lists the
    detected faces (and writes variants/faces_numbered.jpg) so the right numbers can be chosen."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    img = read(native_path(w, base))
    found = faces.detect(img)
    save(v / "faces_numbered.jpg", to_height(faces.numbered(img, found), min(img.shape[0], 1200)))
    for n, (x, y, fw, fh) in enumerate(found, 1):
        log(f"face {n}: x={x} y={y} size={fw}x{fh}")
    if not picks:
        log(f"see {v / 'faces_numbered.jpg'}")
        return None
    bad = [i for i in picks if not 1 <= i <= len(found)]
    if bad:
        raise ValueError(f"face number(s) {bad} not among the {len(found)} detected faces")
    donors = {i: read(native_path(w, n)) for i, n in picks.items()}
    out, notes = faces.mix_heads(img, donors, found)
    for s in notes:
        log(s)
    n = max(existing_variants(w)) + 1
    save(native_path(w, n), out)
    oh, ow = read(photo).shape[:2]
    save(v / f"variant_{n}_{ow}x{oh}.png", resize(out, ow, oh))
    info_path = w / "variants.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"variants": []}
    info["variants"].append({"n": n, "mixed_from": {"base": base, "faces": picks}})
    info_path.write_text(json.dumps(info, indent=2))
    write_overviews(photo)
    log(f"mixed variant {n}")
    return n


def finish(photo: Path, variant: int, full: bool = False, sharpened: bool = True, log=print) -> Path:
    """SeedVR2-7B upscale of the chosen variant, then the final (original size, or 2K for small
    originals; with full=True the final keeps the whole upscale resolution)."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    orig = read(photo)
    h, wd = orig.shape[:2]
    stem, n = photo.stem, variant
    var = read(native_path(w, n))
    vh, vw = var.shape[:2]
    tw, th = final_size(wd, h)
    s = sr_factor((tw, th), (vw, vh))
    log(f"SeedVR2-7B x{s}: {vw}x{vh} -> {vw * s}x{vh * s}")
    big, stats = seedvr2.upscale(var, s, w / "_seedvr2")
    log(f"  {stats['tiles']} tile(s) in {stats['seconds']} s, GPU peak {stats['gpu_peak_mb']} MB")
    shutil.copy(w / "_seedvr2" / "seedvr2.log", w / f"seedvr2_v{n}.log")
    shutil.rmtree(w / "_seedvr2", ignore_errors=True)
    UW = vw * s
    UH = round(UW * h / wd)  # undo Qwen's multiple-of-16 rounding: exact original proportions
    big = resize(big, UW, UH)
    if full:
        tw, th = UW, UH
        final = big
    else:
        save(d / f"{stem}_4_upscaled_v{n}_{UW}x{UH}.png", big)
        final = resize(big, tw, th)
    out = save(d / f"{stem}_2_final_v{n}_{tw}x{th}.png", final)
    if sharpened:
        save(d / f"{stem}_3_final_v{n}_{tw}x{th}_sharpened.png", unsharp(final))
    ref = resize(orig, tw, th)
    if (tw, th) != (wd, h):
        save(d / f"{stem}_1_original_{tw}x{th}.png", ref)
    save(w / f"compare_original_vs_final_v{n}.jpg",
         np.hstack([label(to_height(ref, 900), "original"), label(to_height(final, 900), f"final (variant {n})")]))
    log(f"done: {out}")
    return out


def crop(photo: Path, inset: float = 0.02, corners=None, out: Path | None = None, log=print) -> Path | None:
    """Straighten and crop a photographed print; writes <photo>_cropped.jpg for the next steps."""
    from . import crop as cropmod
    photo = Path(photo).resolve()
    img = read(photo)
    c = np.float32(corners).reshape(4, 2) if corners is not None else cropmod.find_print(img)
    d, v, w = folders(photo)
    if c is None:
        log("no print edges found; pass --corners x1,y1,...,x4,y4 (TL TR BR BL) to crop by hand")
        return None
    save(w / "crop_check.jpg", to_height(cropmod.preview(img, c), min(img.shape[0], 1400)))
    flat = cropmod.straighten(img, c, inset)
    out = Path(out) if out else config.INPUT / f"{photo.stem}_cropped.jpg"
    save(out, flat)
    log(f"corners (TL TR BR BL): {np.round(cropmod.order_corners(c)).astype(int).tolist()}")
    log(f"check the outline in {w / 'crop_check.jpg'}")
    log(f"cropped {flat.shape[1]}x{flat.shape[0]} -> {out}")
    return out
