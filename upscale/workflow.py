"""The photo workflow: variants -> (pick, optionally mix faces) -> finish.

Layout of output/<photo>/ (the top level is the set to flip through, in viewing order):

    <photo>_0_original_WxH.<ext>                     the untouched original
    <photo>_1_original_<final>.png                   the original resized to the final size (if it differs)
    <photo>_2_final_vN_<model>_<final>.png           THE RESULT: original resolution, or 2K if smaller
    <photo>_3_final_vN_<model>_<final>_sharpened.png
    <photo>_4_upscaled_vN_<model>_<big>.png          the full SeedVR2 output (2x the Qwen output; 3-4x for very large originals)
    variants/  variant_0_original_*, variant_N_WxH.png, contact_sheet.jpg, faces.jpg (+ faces_numbered.jpg from mix)
    work/      Qwen's native outputs, variants.json (seeds, photo checksum), SeedVR2 logs, comparisons
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np

from . import config, faces, prompts, qwen, seedvr2
from .images import (final_size, grid, label, qwen_scale, read, resize, save, sr_factor, to_height,
                     unsharp)

SHEET_W = 800


class WorkflowError(RuntimeError):
    pass


def folders(photo: Path) -> tuple[Path, Path, Path]:
    d = config.OUTPUT / photo.stem
    v, w = d / "variants", d / "work"
    for p in (d, v, w):
        p.mkdir(parents=True, exist_ok=True)
    return d, v, w


def native_path(work: Path, n) -> Path:
    return work / f"variant_{n}_native.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_info(work: Path) -> dict:
    p = work / "variants.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"variants": []}


def save_info(work: Path, info: dict) -> None:
    (work / "variants.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def check_same_photo(photo: Path, info: dict) -> None:
    """output/<stem> is keyed by file name; refuse to mix up two different photos with the same name."""
    sha = info.get("photo_sha256")
    if sha and sha != _sha256(photo):
        raise WorkflowError(f"output/{photo.stem}/ belongs to a different photo with the same name "
                            f"({info.get('photo', '?')}). Rename this photo, or delete or move output/{photo.stem}/ "
                            "to start over with it.")


def used_numbers(d: Path) -> set[int]:
    """Every variant number in use: Qwen natives, delivered variants and the variants.json record,
    so a new variant never overwrites one the user kept (even after deleting work files)."""
    nums = set()
    for p in list((d / "work").glob("variant_*_native.png")) + list((d / "variants").glob("variant_*_*.png")):
        m = re.match(r"variant_(\d+)_", p.name)
        if m:
            nums.add(int(m.group(1)))
    nums |= {int(v["n"]) for v in load_info(d / "work").get("variants", []) if str(v.get("n", "")).isdigit()}
    nums.discard(0)
    return nums


def next_number(d: Path) -> int:
    return max(used_numbers(d), default=0) + 1


def variants(photo: Path, refs: list[Path] = (), colorize: bool = False, seeds: list[int] | None = None,
             keep_server: bool = False, log=print) -> Path:
    """Four Qwen restorations (two seeds x 2), delivered at the photo's own size for flipping."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    info = load_info(w)
    check_same_photo(photo, info)
    orig = read(photo)
    h, wd = orig.shape[:2]
    ext = photo.suffix.lower()
    shutil.copy(photo, d / f"{photo.stem}_0_original_{wd}x{h}{ext}")
    shutil.copy(photo, v / f"variant_0_original_{wd}x{h}{ext}")
    scale = qwen_scale(wd, h)
    info.update(photo=photo.name, photo_sha256=_sha256(photo), original_size=[wd, h], qwen_scale=scale)
    used = qwen.gpu_used_mb()
    if used is not None and used > 4000:
        log(f"warning: {used} MB of VRAM already in use; local-image needs ~15.5 GB and may refuse to start")
    seeds = seeds or [random.randrange(1, 2**31) for _ in range(2)]
    prompt = prompts.pick(colorize, bool(refs))
    tmp = w / "_qwen"
    try:
        for seed in seeds:  # 2 batches of 2: one batch of 4 at 2048 px overflows 16 GB and crawls
            log(f"Qwen edit: seed {seed}, scale {scale}")
            made = qwen.edit(prompt, [photo, *refs], tmp, scale, 2, seed)
            for k, p in enumerate(made):  # keep each batch as soon as it exists
                n = next_number(d)
                native = read(p)
                shutil.move(str(p), str(native_path(w, n)))
                save(v / f"variant_{n}_{wd}x{h}.png", resize(native, wd, h))  # exact original proportions
                info["variants"].append({"n": n, "seed": seed, "batch_index": k, "colorize": colorize,
                                         "refs": [Path(r).name for r in refs],
                                         "native_size": [native.shape[1], native.shape[0]]})
                save_info(w, info)
                log(f"  variant {n}")
    finally:
        if not keep_server:
            qwen.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    write_overviews(photo)
    log(f"done: {v}")
    return v


def write_overviews(photo: Path) -> None:
    """contact_sheet.jpg and faces.jpg for the variants that still exist in variants/ (the user may
    delete rejects). faces_numbered.jpg is written by `mix`, for the base it mixes into."""
    d, v, w = folders(photo)
    orig = read(photo)
    h, wd = orig.shape[:2]
    have = [(n, v / f"variant_{n}_{wd}x{h}.png") for n in sorted(used_numbers(d))]
    have = [(n, p) for n, p in have if p.exists()]
    if not have:
        return
    ims = [orig] + [read(p) for _, p in have]
    labels = ["original"] + [f"variant {n}" for n, _ in have]
    th = round(SHEET_W * h / wd)
    save(v / "contact_sheet.jpg", grid([label(resize(im, SHEET_W, th), t) for im, t in zip(ims, labels)], 3))
    found = faces.detect(ims[1])  # faces are found more reliably in a restored variant
    if found:
        largest = sorted(sorted(found, key=lambda f: -f[2] * f[3])[:3])
        save(v / "faces.jpg", faces.strip(ims, labels, faces=largest))


def _native(w: Path, n: int) -> np.ndarray:
    if n < 1:
        raise WorkflowError("variant 0 is the original; pick a restored variant (1, 2, ...)")
    p = native_path(w, n)
    if not p.exists():
        raise WorkflowError(f"variant {n} not found ({p}); run `variants` first, or pick another number")
    return read(p)


def mix(photo: Path, base: int, picks: dict[int, int] | None, log=print) -> int | None:
    """New variant = `base` with face i taken from variant picks[i]. Without picks, only lists the
    detected faces (and writes variants/faces_numbered.jpg) so the right numbers can be chosen."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    info = load_info(w)
    check_same_photo(photo, info)
    img = _native(w, base)
    found = faces.detect(img)
    save(v / "faces_numbered.jpg", to_height(faces.numbered(img, found), min(img.shape[0], 1200)))
    for n, (x, y, fw, fh) in enumerate(found, 1):
        log(f"face {n}: x={x} y={y} size={fw}x{fh}")
    if not picks:
        log(f"see {v / 'faces_numbered.jpg'}")
        return None
    bad = [i for i in picks if not 1 <= i <= len(found)]
    if bad:
        raise WorkflowError(f"face number(s) {bad} not among the {len(found)} detected faces "
                            f"(see {v / 'faces_numbered.jpg'})")
    donors = {i: _native(w, n) for i, n in picks.items()}
    out, notes = faces.mix_heads(img, donors, found)
    for s in notes:
        log(s)
    n = next_number(d)
    save(native_path(w, n), out)
    oh, ow = read(photo).shape[:2]
    save(v / f"variant_{n}_{ow}x{oh}.png", resize(out, ow, oh))
    info["variants"].append({"n": n, "mixed_from": {"base": base, "faces": {str(k): x for k, x in picks.items()}}})
    save_info(w, info)
    write_overviews(photo)
    log(f"mixed variant {n}")
    return n


def _free_gpu(log) -> None:
    """local-image's server (from `variants --keep-server`) holds ~15.5 GB; SeedVR2 needs the card."""
    used = qwen.gpu_used_mb()
    if used is not None and used > 6000:
        log(f"{used} MB of VRAM in use; stopping the local-image server if it is running")
        try:
            qwen.stop()
        except qwen.LocalImageMissing:
            pass
        used = qwen.gpu_used_mb()
        if used is not None and used > 6000:
            log(f"warning: still {used} MB in use by another program; SeedVR2 may run out of VRAM")


def finish(photo: Path, variant: int, full: bool = False, sharpened: bool = True, model: str | None = None,
           log=print) -> Path:
    """SeedVR2 upscale (7B by default) of the chosen variant, then the final: original size, or 2K
    for small originals; with full=True the final keeps the whole upscale resolution."""
    photo = Path(photo).resolve()
    d, v, w = folders(photo)
    check_same_photo(photo, load_info(w))
    orig = read(photo)
    h, wd = orig.shape[:2]
    stem, n = photo.stem, variant
    var = _native(w, n)
    vh, vw = var.shape[:2]
    tw, th = final_size(wd, h)
    s = sr_factor((tw, th), (vw, vh))
    key = model or config.default_upscaler()
    _free_gpu(log)
    log(f"SeedVR2 {key} x{s}: {vw}x{vh} -> {vw * s}x{vh * s}")
    big, stats = seedvr2.upscale(var, s, w / f"seedvr2_v{n}_{key}.log", model=key)
    log(f"  {stats['tiles']} tile(s) of {stats['tile_in']} px in {stats['seconds']} s, "
        f"GPU peak {stats['gpu_peak_mb']} MB" + (f" ({stats['guard']})" if "guard" in stats else ""))
    UW = vw * s
    UH = round(UW * h / wd)  # undo Qwen's multiple-of-16 rounding: exact original proportions
    big = resize(big, UW, UH)
    if full:
        tw, th = UW, UH
        final = big
    else:
        save(d / f"{stem}_4_upscaled_v{n}_{key}_{UW}x{UH}.png", big)
        final = resize(big, tw, th)
    out = save(d / f"{stem}_2_final_v{n}_{key}_{tw}x{th}.png", final)
    if sharpened:
        save(d / f"{stem}_3_final_v{n}_{key}_{tw}x{th}_sharpened.png", unsharp(final))
    ref = resize(orig, tw, th)
    if (tw, th) != (wd, h):
        save(d / f"{stem}_1_original_{tw}x{th}.png", ref)
    save(w / f"compare_original_vs_final_v{n}_{key}.jpg",
         np.hstack([label(to_height(ref, 900), "original"),
                    label(to_height(final, 900), f"final (variant {n}, {key})")]))
    log(f"done: {out}")
    return out


def crop(photo: Path, inset: float = 0.02, corners=None, out: Path | None = None, log=print) -> Path | None:
    """Straighten and crop a photographed print; writes input/<photo>_cropped.jpg for the next steps."""
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
