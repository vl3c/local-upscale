"""local-upscale command line.

    python scripts/upscale.py status
    python scripts/upscale.py crop     input/photo.jpg [--inset 0.02] [--corners x1,y1,...,x4,y4]
    python scripts/upscale.py variants input/photo.jpg [--ref other.jpg ...] [--colorize] [--seeds A B] [--also TEXT | --prompt TEXT]
    python scripts/upscale.py mix      input/photo.jpg --base 3 [--face 1=2 2=4]
    python scripts/upscale.py finish   input/photo.jpg --variant 2 [--full] [--no-sharpened] [--model 7b|7b-sharp|3b]
    python scripts/upscale.py stop
    python scripts/upscale.py compare  input/photo.jpg | output/photo/some_folder

variants and finish use the GPU, so they run in their own visible window (see upscale/window.py):
this command waits for the job and echoes its log. Close that window, or run `stop` (stop.cmd),
to stop the job and free the GPU.

Run with the repo's .venv Python. See README.md and docs/workflow.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upscale import config  # noqa: E402


def parse_picks(items):
    picks = {}
    for it in items or []:
        parts = it.replace(":", "=").split("=")
        if len(parts) != 2 or not all(x.strip().isdigit() for x in parts):
            raise SystemExit(f"--face expects FACE=VARIANT pairs like 1=2, got {it!r}")
        picks[int(parts[0])] = int(parts[1])
    return picks


def parse_corners(s):
    v = [float(x) for x in s.replace(";", ",").split(",")]
    if len(v) != 8:
        raise argparse.ArgumentTypeError("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (TL TR BR BL)")
    return v


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: upscale.py stop reads a job's options from its command line (--keep-server).
    ap = argparse.ArgumentParser(prog="upscale", description="Restore and upscale photos locally.",
                                 allow_abbrev=False)
    ap.add_argument("--in-window", action="store_true", help=argparse.SUPPRESS)  # set by window.run
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **k):
        return sub.add_parser(name, allow_abbrev=False, **k)

    add("status", help="check local-image, the engine, the weights and the GPU")
    add("stop", help="stop a running variants or finish job (as stop.cmd does) and free the GPU")

    c = add("crop", help="straighten and crop a photographed print (before variants)")
    c.add_argument("photo", type=Path)
    c.add_argument("--inset", type=float, default=0.02, help="fraction trimmed from each side (default 0.02)")
    c.add_argument("--corners", type=parse_corners,
                   help="manual corners x1,y1,...,x4,y4 (TL TR BR BL); use --corners=... if one is negative")
    c.add_argument("--out", type=Path, help="default: input/<photo>_cropped.jpg")

    v = add("variants", help="4 cleaned-up versions to choose from (Qwen-Image)")
    v.add_argument("photo", type=Path)
    v.add_argument("--ref", type=Path, nargs="+", default=[], help="other photos of the same people (faces only)")
    v.add_argument("--colorize", action="store_true", help="black-and-white photo: restore and colorize")
    v.add_argument("--seeds", type=int, nargs=2, help="two seeds (2 variants each); random if omitted")
    v.add_argument("--keep-server", action="store_true", help="leave the local-image server running")
    v.add_argument("--prompt", help="replace the restore instruction entirely (refs are passed as given)")
    v.add_argument("--also", help="an extra edit done with the restore, e.g. 'Remove the blue helmet from the seat'")

    m = add("mix", help="take faces from other variants into one (no --face: list the faces)")
    m.add_argument("photo", type=Path)
    m.add_argument("--base", type=int, required=True)
    m.add_argument("--face", nargs="+", help="N=V: face N (left to right) from variant V")

    f = add("finish", help="SeedVR2 upscale of the chosen variant (--model, or the install default), then the final")
    f.add_argument("photo", type=Path)
    f.add_argument("--variant", type=int, required=True)
    f.add_argument("--full", action="store_true", help="keep the full upscale resolution as the final")
    f.add_argument("--no-sharpened", action="store_true", help="skip the lightly sharpened copy")
    f.add_argument("--model", choices=config.upscaler_names(),
                   help="upscaler (default: the one chosen at install, else 7b)")
    k = add("compare", help="(re)write compare.html: flip between the images with keys 1, 2, 3...")
    k.add_argument("target", type=Path, help="a photo (its output folder) or any folder of images")
    return ap


def compare_page(target: Path) -> int:
    from upscale import compare
    target = target.resolve()
    d = target if target.is_dir() else config.OUTPUT / target.stem
    if not d.is_dir():
        print(f"error: no folder {d}; run `variants` first, or pass a folder of images", file=sys.stderr)
        return 1
    out = compare.write(d, prefix=None if target.is_dir() else target.stem)
    if not out:
        print(f"error: no images in {d}", file=sys.stderr)
        return 1
    print(out)
    return 0


def status() -> int:
    from upscale import qwen
    ok = True

    def line(good, text):
        nonlocal ok
        ok &= bool(good)
        print(("  ok   " if good else "  MISSING ") + text)

    err = config.local_config_error()
    if err:
        line(False, err)
    try:
        gen, _ = qwen.paths()
        line(True, f"local-image: {gen.parent.parent}")
    except qwen.LocalImageMissing as e:
        line(False, str(e))
    line((config.ENGINE / "inference_cli.py").exists(), f"SeedVR2 engine: {config.ENGINE}")
    inst = config.installed_upscalers()
    line(bool(inst), f"upscalers installed: {', '.join(inst) or 'none'} (default: {config.default_upscaler()})")
    line(config.default_upscaler() in inst, f"default upscaler weights: {config.upscaler_file()}")
    line((config.SEEDVR2_MODELS / "ema_vae_fp16.safetensors").exists(), "SeedVR2 VAE: ema_vae_fp16.safetensors")
    line(config.FACE_MODEL.exists(), "face detector: models/face/" + config.FACE_MODEL.name)
    used = qwen.gpu_used_mb()
    line(used is not None, "GPU: nvidia-smi" + (f" ({used} MB in use)" if used is not None else " not found"))
    free_gb = shutil.disk_usage(config.ROOT).free / 2**30
    print(f"  info {free_gb:.0f} GB free on the repo's drive")
    return 0 if ok else 1


def stop() -> int:
    from upscale import qwen, window
    code = window.stop() or 0
    used = qwen.gpu_used_mb()
    if used is not None:
        print(f"  VRAM now in use: {used} MB")
    return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")  # a piped stdout is cp1252: photo names may not be
    except (AttributeError, ValueError):
        pass
    if a.cmd == "status":
        return status()
    if a.cmd == "stop":
        return stop()
    if a.cmd == "compare":
        return compare_page(a.target)
    from upscale import window
    if a.cmd in window.WINDOWED:
        if not a.in_window and os.name == "nt":
            missing = [p for p in [a.photo, *getattr(a, "ref", [])] if not p.is_file()]
            if missing:  # don't open a window only to show this for a minute
                print(f"error: no such file: {', '.join(map(str, missing))}", file=sys.stderr)
                return 1
            return window.run(argv, a.cmd, a.photo)
        if a.in_window:
            print(f"  local-upscale {a.cmd} {a.photo.name}\n"
                  "  Close this window (or run stop.cmd) to stop it and free the GPU.\n", flush=True)
    from upscale import qwen, seedvr2, workflow
    expected = (FileNotFoundError, ValueError, workflow.WorkflowError, seedvr2.SeedVR2Error,
                qwen.LocalImageMissing, qwen.QwenError)
    try:
        if a.cmd == "crop":
            return 0 if workflow.crop(a.photo, a.inset, a.corners, a.out) else 1
        if a.cmd == "variants":
            workflow.variants(a.photo, a.ref, a.colorize, a.seeds, a.keep_server, also=a.also,
                              prompt=a.prompt)
        elif a.cmd == "mix":
            workflow.mix(a.photo, a.base, parse_picks(a.face))
        elif a.cmd == "finish":
            workflow.finish(a.photo, a.variant, full=a.full, sharpened=not a.no_sharpened, model=a.model)
    except expected as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
