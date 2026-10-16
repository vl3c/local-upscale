"""Fetch the pinned weights listed in models.json, verifying every SHA-256. Stdlib only.

    python scripts/download_models.py
    python scripts/download_models.py --reuse D:\\other\\models     # hard-link matching files instead

Re-running is safe: verified files are skipped and partial downloads resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
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
    req = urllib.request.Request(url, headers={"User-Agent": "local-upscale"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", type=Path, nargs="*", default=[], help="folders to take matching weights from")
    a = ap.parse_args()
    manifest = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    bad = 0
    for f in manifest["files"]:
        dest = ROOT / f["dest"] / f["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {f['name']} ({f['bytes'] / 2**30:.2f} GB, {f['license']})")
        if dest.exists():
            if dest.stat().st_size == f["bytes"] and sha256(dest) == f["sha256"]:
                print("    ok")
                continue
            dest.unlink()
        if not reuse(f["name"], f["sha256"], a.reuse, dest):
            download(f["url"], dest, f["bytes"])
        if sha256(dest) != f["sha256"]:
            print("    CHECKSUM MISMATCH, deleted")
            dest.unlink()
            bad += 1
        else:
            print("    verified")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
