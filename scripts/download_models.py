"""Fetch the pinned weights listed in models.json, verifying every SHA-256. Stdlib only.

    python scripts/download_models.py                        # what's installed (fresh install: 7b) + shared files
    python scripts/download_models.py --upscaler 7b 3b       # several; the first becomes the default
    python scripts/download_models.py --list                 # show the choices
    python scripts/download_models.py --reuse D:\\other\\models  # hard-link matching files instead

The first --upscaler is saved as the default in config.local.json (used by `finish`).
Re-running is safe: verified files are skipped and partial downloads resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have >= size:  # complete (or oversized) leftover: let the checksum decide
        os.replace(part, dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "local-upscale"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable: the partial file is unusable, start over
            part.unlink(missing_ok=True)
            return download(url, dest, size)
        raise
    with r, open(part, "ab" if have and r.status == 206 else "wb") as f:
        done = have if r.status == 206 else 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r    {done / 2**20:,.0f} / {size / 2**20:,.0f} MB", end="", flush=True)
    print()
    os.replace(part, dest)


def reuse(name: str, sha: str, dirs: list[Path], dest: Path) -> bool:
    for d in dirs:
        for cand in d.rglob(name) if d.exists() else []:
            try:
                os.link(cand, dest)          # hard link: no extra disk space (same drive)
            except OSError:
                shutil.copy2(cand, dest)     # across drives
            if sha256(dest) == sha:
                print(f"    reused {cand}")
                return True
            dest.unlink()
    return False


def fetch(f: dict, reuse_dirs: list[Path]) -> bool:
    dest = ROOT / f["dest"] / f["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {f['name']} ({f['bytes'] / 2**30:.2f} GB, {f['license']})")
    if dest.exists():
        if dest.stat().st_size == f["bytes"] and sha256(dest) == f["sha256"]:
            print("    ok")
            return True
        dest.unlink()
    if not reuse(f["name"], f["sha256"], reuse_dirs, dest):
        download(f["url"], dest, f["bytes"])
    if sha256(dest) != f["sha256"]:
        print("    CHECKSUM MISMATCH, deleted")
        dest.unlink()
        return False
    print("    verified")
    return True


def local_config() -> dict | None:
    """config.local.json as a dict ({} if absent), or None if it exists but isn't valid JSON."""
    p = ROOT / "config.local.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None


def save_default(name: str) -> None:
    cfg = local_config()
    if cfg is None:
        print("  config.local.json is not valid JSON; not changing it (fix it, then re-run)")
        return
    cfg["upscaler"] = name
    (ROOT / "config.local.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def installed(ups: dict) -> list[str]:
    return [k for k, v in ups.items() if (ROOT / v["dest"] / v["name"]).exists()]


def main() -> int:
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    ups = manifest["upscalers"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--upscaler", nargs="+", choices=list(ups),
                    help="which SeedVR2 upscaler(s) to fetch; the first becomes the default. Without it: "
                         f"the installed ones, else config.local.json's choice, else {manifest['default_upscaler']}")
    ap.add_argument("--list", action="store_true", help="list the upscalers and exit")
    ap.add_argument("--reuse", type=Path, nargs="*", default=[], help="folders to take matching weights from")
    a = ap.parse_args()
    if a.list:
        for k, v in ups.items():
            print(f"  {k:9s} {v['bytes'] / 2**30:5.1f} GB  {v['note']}")
        return 0
    # Without --upscaler (e.g. re-running install.ps1): keep what is installed or was chosen; only a
    # fresh install gets the default. Never pull a model the user didn't pick.
    cfg = local_config() or {}
    chosen = (a.upscaler or installed(ups)
              or [cfg["upscaler"] if cfg.get("upscaler") in ups else manifest["default_upscaler"]])
    ok = all([fetch(ups[k], a.reuse) for k in chosen] + [fetch(f, a.reuse) for f in manifest["shared"]])
    if ok and a.upscaler:
        save_default(chosen[0])
        print(f"  default upscaler: {chosen[0]} (config.local.json)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
