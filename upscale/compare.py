"""compare.html: a local page to flip between the original, the variants and the finals.

Every image is shown at the same size and in the same place; the number keys 1, 2, 3 ... (or the
buttons) switch between them, holding the photo shows the original, Space switches to it (a tap)
or shows it while held, Z zooms to full size. The page links the images next to it by relative path, so it works opened straight from disk,
and loads nothing from the network.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from .images import read, save

PAGE = "compare.html"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".avif"}  # what browsers show
PREVIEWED = {".tif", ".tiff"}  # an original in these gets a PNG copy the page can show
OVERVIEWS = {"contact_sheet", "faces", "faces_numbered"}  # side-by-side sheets, not comparable images
KEYS = list("1234567890") + list("qwertyuiop")
_ORIGINAL = re.compile(r"(^|_)0_original(_|$)|^original$", re.I)
_SIZE = re.compile(r"_?\d+x\d+(?=_|$)")


def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.stem.lower() not in OVERVIEWS and (
        p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() in PREVIEWED and is_original(p))]


def is_original(p: Path) -> bool:
    return bool(_ORIGINAL.search(p.stem))


def collect(d: Path) -> list[Path]:
    """The images worth flipping through in `d`, in viewing order: the original, then variants/
    (by number), then the rest of `d` (finals sort 1_original, 2_final, 3_sharpened, 4_upscaled).
    A folder with no original of its own (an ad-hoc set inside a photo's folder, say) borrows the
    parent folder's. The original sits both in `d` and in variants/; byte-identical copies appear once."""
    d = Path(d).resolve()
    top, var = _images(d), _images(d / "variants")
    found = top + var
    if not any(is_original(p) for p in found):
        found += [p for p in _images(d.parent) if is_original(p)][:1]

    def key(p: Path):
        group = 0 if is_original(p) else 1 if p.parent == d / "variants" else 2
        return group, _natural(p.name)

    out, seen = [], set()
    for p in sorted(found, key=key):
        h = is_original(p) and hashlib.sha256(p.read_bytes()).hexdigest()
        if not h or h not in seen:
            seen.add(h)
            out.append(p)
    return out


def label(p: Path, prefix: str = "") -> str:
    """A short button label from a file name: "variant_3_1448x1878" -> "Variant 3",
    "<photo>_2_final_v3_7b_2896x3756" -> "Final v3 7b", "<photo>_0_original_..." -> "Original"."""
    s = p.stem
    if is_original(p):
        return "Original"
    if prefix and s.lower().startswith(prefix.lower() + "_"):
        s = s[len(prefix) + 1:]
    s = re.sub(r"^\d+_(?=\D)", "", s)  # the viewing-order index
    s = _SIZE.sub("", s).replace("_", " ").strip() or p.stem
    return s[:1].upper() + s[1:]


def _src(p: Path, d: Path) -> str:
    try:
        rel = p.resolve().relative_to(d.resolve())
    except ValueError:  # the parent's original: one level up
        rel = Path("..") / p.resolve().relative_to(d.resolve().parent)
    return quote(rel.as_posix())


def _viewable(p: Path, d: Path) -> Path:
    """`p`, or for a TIFF original a PNG copy of it in `d`/work/ (browsers can't show TIFF)."""
    if p.suffix.lower() in IMAGE_EXTS:
        return p
    out = d / "work" / "compare_original.png"
    if not out.exists() or out.stat().st_mtime < p.stat().st_mtime:
        save(out, read(p))
    return out


def write(d: Path, title: str | None = None, prefix: str | None = None) -> Path | None:
    """Write `d`/compare.html for the images in `d` (see `collect`). Returns None if there are none.
    `prefix` (the photo's name) is trimmed from labels; it defaults to the folder's name."""
    d = Path(d).resolve()
    items = collect(d)
    if not items:
        return None
    prefix = d.name if prefix is None else prefix
    labels = [label(p, prefix) for p in items]
    for i, t in enumerate(labels):  # e.g. the original, then the original resized to the final size
        if t in labels[:i]:
            size = _SIZE.search(items[i].stem)
            labels[i] = f"{t} {size.group(0).lstrip('_')}" if size else f"{t} ({items[i].name})"
    data = [{"src": _src(_viewable(p, d), d), "label": t, "file": p.name, "original": is_original(p),
             "key": KEYS[i] if i < len(KEYS) else ""} for i, (p, t) in enumerate(zip(items, labels))]
    fill = {"TITLE": html.escape(title or d.name), "DATA": json.dumps(data).replace("<", "\\u003c")}
    page = re.sub(r"\{(TITLE|DATA)\}", lambda m: fill[m.group(1)], TEMPLATE)
    out = d / PAGE
    out.write_text(page, encoding="utf-8")
    return out


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}: compare</title>
<style>
:root{--bg:#121116;--surface:#1d1b22;--fg:#ece9f1;--muted:#9a95a5;--line:#34313d;
  --accent:#f0679a;--accent-ink:#1a0f14;color-scheme:dark}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;
  display:flex;flex-direction:column;overflow:hidden}
header{display:flex;flex-direction:column;gap:8px;padding:8px 16px 10px}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px}
h1{font-size:16px;font-weight:600;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40ch}
.tabs{display:flex;flex-wrap:wrap;justify-content:center;gap:8px}
.tabs button{font:600 19.5px system-ui,sans-serif;border:1px solid var(--line);background:var(--surface);color:var(--fg);
  border-radius:999px;padding:6px 18px 6px 8px;cursor:pointer;display:flex;align-items:center;gap:10px}
.tabs button kbd{font:600 18px ui-monospace,Consolas,monospace;min-width:30px;height:30px;display:grid;place-items:center;
  border-radius:999px;background:var(--line);color:var(--fg)}
.tabs button.peek{border-color:var(--accent)}
.tabs button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.tabs button[aria-pressed="true"] kbd{background:var(--accent-ink);color:var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
.hint{color:var(--muted);font-size:12.5px;margin-left:auto}
.stage{flex:1;min-height:0;overflow:hidden;display:grid;place-items:center;padding:0 16px 12px}
.stage.zoom{overflow:auto;place-items:start;padding:0}
.viewer{position:relative;user-select:none;-webkit-user-select:none;touch-action:manipulation;cursor:zoom-in}
.zoom .viewer{cursor:grab;margin:auto}
.viewer img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;visibility:hidden;pointer-events:none}
.viewer img.on{visibility:visible}
.tag{position:fixed;left:24px;bottom:22px;background:rgba(0,0,0,.65);color:#fff;font:600 13px system-ui,sans-serif;
  padding:4px 10px;border-radius:999px;pointer-events:none}
</style>
</head>
<body>
<header>
  <div class="top">
    <h1>{TITLE}</h1>
    <span class="hint">1, 2, 3&hellip; or &larr; &rarr; switch<span id="holdhint"> &middot; Space: original (tap to switch, hold to peek) &middot; hold the photo: original</span> &middot; Z or double-click: full size</span>
  </div>
  <div class="tabs" id="tabs" role="group" aria-label="Choose image"></div>
</header>
<div class="stage" id="stage"><div class="viewer" id="viewer"></div></div>
<span class="tag" id="tag"></span>
<script>
(function(){
  var ITEMS = {DATA};
  var tabs = document.getElementById('tabs'), viewer = document.getElementById('viewer');
  var stage = document.getElementById('stage'), tag = document.getElementById('tag');
  var imgs = [], btns = [], cur = 0, shown = 0, zoom = false, peek = false, peekAt = 0;
  var orig = ITEMS.findIndex(function(it){ return it.original; });
  if (orig < 0) document.getElementById('holdhint').hidden = true;  // nothing to hold for
  ITEMS.forEach(function(it, i){
    var im = new Image(); im.src = it.src; im.alt = it.label; im.decoding = 'sync';
    im.addEventListener('load', layout);
    viewer.appendChild(im); imgs.push(im);
    var b = document.createElement('button');
    b.type = 'button'; b.title = it.file;
    if (it.key) { var k = document.createElement('kbd'); k.textContent = it.key.toUpperCase(); b.appendChild(k); }
    b.appendChild(document.createTextNode(it.label));
    b.addEventListener('click', function(){ pick(i); b.blur(); });
    tabs.appendChild(b); btns.push(b);
  });
  var bw = Math.max.apply(null, btns.map(function(b){ return b.offsetWidth; }));
  btns.forEach(function(b){ b.style.minWidth = bw + 'px'; b.style.justifyContent = 'center'; });
  function show(i){
    shown = i;
    btns.forEach(function(b, k){ b.classList.toggle('peek', k === i && k !== cur); });
    imgs.forEach(function(im, k){ im.classList.toggle('on', k === i); });
    var im = imgs[i];
    tag.textContent = ITEMS[i].label + (im.naturalWidth ? '  ' + im.naturalWidth + ' \\u00d7 ' + im.naturalHeight : '');
  }
  function pick(i){
    cur = i; peek = false;
    btns.forEach(function(b, k){ b.setAttribute('aria-pressed', k === i ? 'true' : 'false'); });
    show(i);
    try { history.replaceState(null, '', '#' + (i + 1)); } catch (e) {}
  }
  function ratio(){  // the original's proportions, else the first image that has loaded
    var order = (orig < 0 ? [] : [orig]).concat(imgs.map(function(_, k){ return k; }));
    for (var j = 0; j < order.length; j++) { var im = imgs[order[j]]; if (im.naturalWidth) return im.naturalWidth / im.naturalHeight; }
    return 0;
  }
  function layout(){
    var r = ratio(); if (!r) return;
    var w, h;
    if (zoom) {  // every image at the size of the largest, so they still line up
      w = Math.max.apply(null, imgs.map(function(im){ return im.naturalWidth || 0; })); h = w / r;
    } else {
      var aw = stage.clientWidth - 32, ah = stage.clientHeight - 12;
      w = Math.min(aw, ah * r); h = w / r;
    }
    viewer.style.width = Math.round(w) + 'px'; viewer.style.height = Math.round(h) + 'px';
    show(shown);
  }
  function setZoom(on, e){
    var fx = .5, fy = .5;
    if (e) { var b = viewer.getBoundingClientRect(); fx = (e.clientX - b.left) / b.width; fy = (e.clientY - b.top) / b.height; }
    zoom = on; stage.classList.toggle('zoom', on); layout();
    if (on) {
      stage.scrollLeft = fx * viewer.offsetWidth - stage.clientWidth / 2;
      stage.scrollTop = fy * viewer.offsetHeight - stage.clientHeight / 2;
    }
  }
  // Hold the photo: the original. In full size, dragging pans instead.
  var down = null;
  viewer.addEventListener('pointerdown', function(e){
    if (e.button !== 0) return;
    down = {x: e.clientX, y: e.clientY, sl: stage.scrollLeft, st: stage.scrollTop, pan: false};
    viewer.setPointerCapture(e.pointerId); if (orig >= 0) show(orig);
  });
  viewer.addEventListener('pointermove', function(e){
    if (!down || !zoom) return;
    var dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (!down.pan && Math.abs(dx) + Math.abs(dy) > 4) { down.pan = true; show(cur); }
    if (down.pan) { stage.scrollLeft = down.sl - dx; stage.scrollTop = down.st - dy; }
  });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function(t){
    viewer.addEventListener(t, function(){ if (down) { down = null; peek = false; show(cur); } });
  });
  viewer.addEventListener('dblclick', function(e){ setZoom(!zoom, e); });
  viewer.addEventListener('contextmenu', function(e){ e.preventDefault(); });
  document.addEventListener('keydown', function(e){
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    var k = e.key.toLowerCase(), n = ITEMS.length;
    var i = ITEMS.findIndex(function(it){ return it.key === k; });
    if (i >= 0) pick(i);
    else if (k === 'arrowright') pick((cur + 1) % n);
    else if (k === 'arrowleft') pick((cur + n - 1) % n);
    else if (k === ' ') {  // a tap switches to the original and back; a hold shows it until released
      if (!e.repeat && orig >= 0) { peek = !peek; peekAt = Date.now(); show(peek ? orig : cur); }
    }
    else if (k === 'z') setZoom(!zoom);
    else if (k === 'escape' && zoom) setZoom(false);
    else return;
    e.preventDefault();
  });
  document.addEventListener('keyup', function(e){
    if (e.key === ' ' && peek && Date.now() - peekAt > 300) { peek = false; show(cur); }
  });
  window.addEventListener('blur', function(){ peek = false; show(cur); });
  window.addEventListener('resize', layout);
  var start = parseInt((location.hash || '').slice(1), 10) - 1;
  pick(start >= 0 && start < ITEMS.length ? start : Math.min(1, ITEMS.length - 1));
  layout();
})();
</script>
</body>
</html>
"""
