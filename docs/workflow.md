# The workflow in depth

This tool came out of a long comparison of local upscalers on real, difficult photos: phone and
webcam pictures from the 2000s, low-light shots, colour casts, prints photographed on a table, and
black-and-white prints. This page records what the pipeline does, why, and what was tried and dropped.

## Pipeline

```
photo ──► [crop]  ──► variants (Qwen-Image edit, 4x) ──► you pick ──► [mix faces] ──► finish
           prints       clean-up / colorize                              SeedVR2 x2..4    ─► area downscale
```

### 1. Variants: Qwen-Image instruction edit

- The photo is sent to local-image (Qwen-Image-2.1 on ComfyUI) as an **edit** with an instruction to
  restore it: remove blur, noise and JPEG artefacts, fix the colour cast, and keep every face,
  expression, hair, clothing and the scene exactly as they are. The prompts are in
  `upscale/prompts.py`.
- **Size:** Qwen renders at about the photo's own size, between 1024 and 2048 px (its maximum is
  ~4.2 MP). Smaller photos are rendered at 1024 at least, so faces get enough pixels.
- **4 variants** come from two seeds, 2 images each. A single batch of 4 at 2048 px overflows 16 GB
  of VRAM and slows to a crawl, so it's split.
- **Proportions:** Qwen works in multiples of 16 px and stretches the picture slightly. Every
  delivered image is resized back to the original's exact aspect ratio (residual misalignment
  measured at about 0.1 px).
- The local-image server holds about 15.5 GB of VRAM, so it is stopped after this step to make room
  for SeedVR2.

**Colorize** swaps in a prompt that also repairs scratches, dust and fading and adds natural,
era-appropriate colour. Colours of clothes and objects are the model's guess; variants differ mostly
there.

**References (`--ref`)** add other photos of the same people as extra images in the edit. The prompt
says to use them only for facial features and never copy their background, clothes, pose or
lighting. Without that clause Qwen sometimes returns a reference photo wholesale. This made the
difference on a 563x480 webcam photo where faces were ~70 px tall: with a second frame from the same
evening as a reference, the restored faces finally looked right. Photos from much later in life used
as references *changed* the faces instead.

### 2. Picking, and mixing faces

People judge familiar faces with extreme sensitivity: a millimetre of difference in an eyelid or a
mouth corner reads as "not them". Every seed draws faces slightly differently, so the variants are
compared by face. `variants/faces.jpg` shows up to three faces side by side across all variants.

`mix` combines variants when each person looks best in a different one. Faces are numbered left to
right (`mix` without `--face` writes `variants/faces_numbered.jpg`; check it, since the detector can
also box a hand or a printed face on a shirt). For each replaced face:
- the head region (face and hair) is taken from the donor variant;
- it is aligned by phase correlation (donors come from the same input, so offsets are about 1 px);
- it is colour-matched in LAB to the base and feather-blended;
- nothing below the chin is replaced, so collars and clothes always stay consistent.

### 3. Finish: SeedVR2 upscale and downscale

- **SeedVR2** is a one-step diffusion super-resolution model. It was built for video, so a still is
  cut into square tiles, written as a short lossless FFV1 video, and run through its own CLI with
  batch size 1. The tiles are then feather-blended back.
- **1024 px tiles with 128 px overlap.** Small tiles (448 px were tried) give the model too little
  context: faces come out soft and grainy. No seams were visible with 1024 px tiles.
- **7B mixed FP8** (`seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16`): the most faithful SeedVR2
  variant. It keeps the geometry of its input, so the chosen face survives while fine detail is
  added. This is the default. The plain 7B FP8 file is known to produce artefacts; the "sharp" 7B variant is crisper but
  less faithful. Other upscalers can be installed alongside (`install.ps1 -Upscaler 7b,3b`) and picked
  per run with `finish --model`: `7b-sharp` (crisper) or `3b` (about 40% faster, ~10 GB VRAM, invents
  more). The model's name is part of every output file name, so results from different models never
  overwrite each other.
- **Factor:** 2x for almost every photo. Only originals wider than ~4700 px (where 2x of Qwen's ~2368 px output
  would fall short of the original size) get 3x or 4x, with smaller tiles (682 / 512 px; at 512 px faces have less context, but such large originals
  rarely have small faces) so each upscaled tile stays
  at 2048 px and fits in 16 GB; beyond 4x (originals over ~9500 px) the rest is a plain Lanczos resize.
- **Final size:** the original resolution, or 2K on the long side for smaller originals (`--full`
  keeps the whole upscale). The downscale uses area averaging, which keeps the detail and removes
  generative grain.
- **Spill guard:** on Windows, running out of VRAM doesn't fail; the driver silently pages into
  system RAM and everything becomes tens of times slower. The SeedVR2 process is watched and
  stopped if VRAM stays pinned at the limit while the GPU draws little power.

### The job window

`variants` and `finish` run in their own console window, the way local-image runs its server, so
nothing holds the GPU out of sight (`upscale/window.py`). The command you type relaunches itself as
`upscale.py --in-window ...` in a new console, through `scripts/logtee.py` (local-image's, plus a
window title and a one-minute hold after a failure), and then echoes the job's log
(`logs/<time>_<command>_<photo>.log`) until logtee records the exit code.

- **Closing the window** stops the job: Windows ends every process attached to the console, which
  covers the job, the SeedVR2 engine and local-image's client. local-image's server runs in its
  own window, so `variants` also registers a console close handler that starts local-image's
  stop.cmd outside the closing console (Windows allows a handler about 5 seconds).
- **`stop.cmd`** (`upscale.py stop`) finds this install's job windows by their command line
  (junctions resolved), writes `<log>.stop` so the window closes at once instead of holding,
  kills the job's process tree, and stops local-image's server if the job was `variants`.
- **One job at a time:** the GPU can't hold two, so a second `variants` or `finish` refuses to
  start while another job's window still has a running job. A window held open after a failure
  doesn't count.
- SeedVR2's own output still goes to `work/seedvr2_*.log`; its phases and tiles are echoed into
  the window as progress.

### Crop (prints)

For photos of prints lying on a table, or scans showing the scanner lid: `crop` finds the print's
straight edges (Canny plus Hough lines, the outermost long edge on each side), straightens it with a
perspective transform, and trims 2% per side to drop the white paper border. Run it before
`variants`. It writes a preview of the detected outline to `work/crop_check.jpg`; look at it. On an
ordinary photo with a strong straight line near the border it can produce a wrong crop. Pass
`--corners` to crop by hand.

## Measured on an RTX 5080 (16 GB)

| Step | Time | Peak VRAM |
|---|---|---|
| 4 variants, 2592x1944 photo (Qwen at 2048) | ~8 min | ~15.5 GB |
| 4 variants, ~600x480 photo (Qwen at 1024) | ~3 min | ~15 GB |
| Finish, 2368x1760 Qwen output x2 (6 tiles) | ~75 s | ~12-13 GB |
| Finish, 1120x960 x2 (2 tiles), 7B / 3B | ~37 s / ~21 s | ~13 GB / ~10 GB |

## Tried and dropped

| Approach | What happened |
|---|---|
| Real-ESRGAN (general-x4v3) | Fast, but smears detail ("oily"); kills fine texture in photos |
| Nomos2 RealPLKSR | Close to the original on large clean photos; blurry inputs stay blurry, and it turns compression artefacts into grain |
| RealBasicVSR (tested on video) | Painterly, over-processed look |
| ReCaVSR (tested on video) | Soft, loses fine texture |
| SeedVR2 alone (no Qwen step) | Sharper, but keeps colour casts and much of the blur; the Qwen clean-up first is what transforms old photos |
| SeedVR2 3B | Faster, but less faithful than 7B (changes expressions, invents more) |
| local-image's "Enhance" (partial re-render) | Keeps the colour cast and blur, thins faces; on an already restored image it adds harsh contrast |
| Face-by-face redo (each face cropped and restored alone) | Crops lose context; each attempt invents a different face |
| References from other years | Changes the face toward the reference, or copies the reference photo |
| Very large outputs (6x to "16K") | Only magnifies sensor noise; the detail isn't in the source. 2x is the sweet spot, then downscale |
| A 1:1 face "copy" pass (original as a second reference, "match the face exactly") | No real change; at 2048 px with two images it overflowed VRAM |

Small, blurry background faces in group photos are the least reliable part: each variant re-invents
them differently. Whether that matters depends on the photo; for some it does.
