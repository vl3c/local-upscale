# local-upscale

Restore and upscale old photos on your own GPU. You give it a blurry, noisy, colour-cast or
black-and-white photo; it gives back a clean, sharp version at its original size (or at 2K, if the
original was small), with the faces still looking like the people in it. Nothing leaves your machine.

This is a tool for **still photos**. It is not a video or VR upscaler.

## How it works

The pipeline came out of side-by-side testing of many upscalers on real family photos (see
[docs/workflow.md](docs/workflow.md) for what was tried and why this combination won):

1. **Clean-up: Qwen-Image instruction edit.** The photo goes through
   [local-image](https://github.com/vl3c/local-image), a separate local install of Qwen-Image-2.1 on
   ComfyUI, with an instruction like "restore this into a sharp, clean photograph, fix the colour
   cast, keep every face exactly as it is". It removes blur, noise, JPEG blocks and colour casts and
   redraws fine detail. This is where most of the improvement comes from. Faces come out slightly
   different with every random seed, so you get **4 variants** to choose from.
2. **You pick** the variant where the people look most like themselves. Optionally, you can **mix**
   faces: keep one variant and take a particular person's face from another.
3. **Upscale: SeedVR2-7B.** The chosen variant is upscaled 2x (more for small photos) by SeedVR2, a
   one-step diffusion super-resolution model. It works in overlapping 1024 px tiles, fully on the
   GPU. It keeps the geometry, so the face you picked stays the same, while it adds fine detail
   (eyelashes, hair strands, skin, fabric).
4. **Downscale** with area averaging to the original resolution, or to 2K when the original is
   smaller. Rendering at 2x and averaging down keeps the detail and removes generative grain.

There's also **colorize** mode for black-and-white photos, **identity references** (other photos of
the same people, which help a lot when faces are tiny), and **crop** for photographed or scanned
prints.

## Requirements

- **Windows 10/11** and an **NVIDIA GPU with 16 GB of VRAM.** Developed on an RTX 5080. Both models
  run fully in VRAM, with no CPU offload. local-image needs about 15.5 GB and SeedVR2-7B about 14 GB,
  one at a time; the tool stops one before starting the other.
- **[local-image](https://github.com/vl3c/local-image)** installed and working. It brings Qwen-Image
  and ComfyUI (~30 GB). By default it is expected at `~\.claude\skills\local-image`; otherwise set
  `LOCAL_IMAGE_DIR` or put `{"local_image_dir": "D:\\path\\to\\local-image"}` in `config.local.json`.
- **Python 3.13** (with the `py` launcher) and **git**.
- About **20 GB** of disk: SeedVR2-7B weights ~9 GB, the venv ~6 GB.

## Install

```powershell
git clone https://github.com/vl3c/local-upscale.git
cd local-upscale
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer:
- clones the SeedVR2 CLI at a pinned commit into `engine\`;
- builds `.venv\` (PyTorch from the CUDA 13.0 wheel index);
- downloads the SeedVR2-7B weights and a small face detector into `models\`, checksum-verified;
- links the folder into `~\.claude\skills\local-upscale` for Claude Code (skip with `-NoSkillLink`).

If you already have the SeedVR2 weights somewhere, use `-ReuseModels <folder>` to hard-link them
instead of downloading. Re-running the installer is safe. Check the setup at any time with:

```powershell
.venv\Scripts\python.exe scripts\upscale.py status
```

## Usage

Put photos in `input\` (any folder works; `input\` and `output\` are ignored by git). Each photo
gets its own folder in `output\`.

```powershell
# 1. Four cleaned-up variants to choose from (~3-8 min)
.venv\Scripts\python.exe scripts\upscale.py variants input\photo.jpg

# 2. Look at output\photo\variants\ (flip through variant_0_original, variant_1, ...;
#    contact_sheet.jpg and faces.jpg show them side by side), then finish your pick (~1-2 min)
.venv\Scripts\python.exe scripts\upscale.py finish input\photo.jpg --variant 3
```

Options and extra steps (`upscale.py` below means `.venv\Scripts\python.exe scripts\upscale.py`):

```powershell
upscale.py variants input\photo.jpg --colorize                 # black-and-white photo
upscale.py variants input\photo.jpg --ref input\same_day.jpg   # tiny faces: other photos of the same people

# Mix faces: keep variant 3, take face 1 (counted left to right) from variant 2.
# Without --face it lists the faces and writes variants\faces_numbered.jpg.
upscale.py mix input\photo.jpg --base 3 --face 1=2
upscale.py finish input\photo.jpg --variant 5               # the mixed result is the next variant number

# Photographed or scanned print: straighten and crop it first, then work on the cropped file.
upscale.py crop input\scan.jpg     # writes input\scan_cropped.jpg; check output\scan\work\crop_check.jpg
upscale.py variants input\scan_cropped.jpg --colorize

# Keep the full upscale resolution as the final instead of downscaling.
upscale.py finish input\photo.jpg --variant 3 --full
```

### What you get (`output\<photo>\`)

The top level holds only the images worth comparing, named so they sort in viewing order and can be
flipped through in any image viewer:

| File | |
|---|---|
| `<photo>_0_original_WxH.jpg` | the untouched original |
| `<photo>_1_original_<final>.png` | the original enlarged to the final size (small originals only) |
| `<photo>_2_final_vN_<final>.png` | **the result**: original resolution, or 2K |
| `<photo>_3_final_vN_<final>_sharpened.png` | the same with a light unsharp mask |
| `<photo>_4_upscaled_vN_<big>.png` | the full SeedVR2 output (about 2x the final) |

- `variants\` holds the candidates at the original size, the original as `variant_0_original_*`,
  `contact_sheet.jpg` and `faces.jpg` (close-ups of up to three faces across all variants). Delete
  the ones you don't want; the overviews only show what's left.
- `work\` holds the intermediates: Qwen's full-resolution outputs, `variants.json` with the seeds
  (to reproduce any variant), and SeedVR2 logs.

Every delivered image has exactly the original's proportions.

## Good to know

- **Faces are the hard part.** People are extraordinarily sensitive to small differences in a familiar
  face. That's why there are 4 variants, the `mix` step and `--ref`. When a face is only a few dozen
  pixels tall, the model has to invent most of it. Another photo of the same person from the same
  day or period (`--ref`) helps more than anything else. References from many years apart change
  the face instead.
- **It invents detail.** Both models are generative. Text, small patterns and background faces can be
  plausible rather than true. Don't use it where exactness matters.
- **Don't push the size.** 2x (up to 4x for small originals) is the sweet spot. Going much further (to
  "16K", for example) only magnifies the source's noise; the information isn't there.
- **Colours in colorize mode are guesses.** They're natural and era-appropriate, but the model can't
  know the real colour of a coat. Generate variants and pick.

## Privacy

Everything runs locally. Your photos are read from wherever you point the tool and written only under
`output\`. local-image also keeps a copy of each photo it edits in its own `ComfyUI\input\` folder.
`input\`, `output\`, every image and video file type, the weights and the environments are all
git-ignored. `scripts\check_repo_safe.py` (run before any push) refuses to publish images, weights,
local paths, emails or credentials.

## Licensing

- **This repo's code:** MIT ([LICENSE](LICENSE)).
- **Qwen-Image-2.1** (used through local-image): **Qwen Research License, for research and evaluation
  only.** Personal testing is fine. Commercial use of the results needs a licence from Qwen.
- **SeedVR2** code and weights: Apache-2.0. **YuNet** face detector: MIT.

See [LICENSE-MODELS.md](LICENSE-MODELS.md).

## Repo layout

```
upscale/            the package: workflow, Qwen (local-image) wrapper, SeedVR2 tiling, faces, crop
scripts/upscale.py  the command line
scripts/download_models.py   pinned, checksum-verified weight downloads (models.json)
scripts/check_repo_safe.py   pre-publish safety check
SKILL.md            instructions for Claude Code (the local-upscale skill)
docs/workflow.md    the workflow in depth, and what was tried and rejected
tests/              run without a GPU: python -m pytest
install.ps1         setup
```

Created on install and ignored by git: `engine\` (the SeedVR2 CLI), `models\`, `.venv\`, `.cache\`,
`input\`, `output\`.
