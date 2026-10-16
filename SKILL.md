---
name: local-upscale
description: Restore, clean up, colorize and upscale old photos locally on this machine's GPU: removes blur, noise, JPEG artefacts and colour casts with a Qwen-Image instruction edit (through the local-image tool), offers 4 variants to choose from (faces differ per variant), can mix faces between variants and use other photos of the same people as identity references, then upscales the pick with SeedVR2 (7B by default; 7B-sharp or 3B if installed) and delivers it at the original resolution (or 2K for small originals). Also colorizes black-and-white photos and straightens/crops photographed prints. Use when the user asks to restore, enhance, sharpen, fix, colorize or upscale a photo or picture locally. Not for video, and not for generating new images (use local-image for that).
---

# local-upscale: restore and upscale photos on the local GPU

The user gives a photo; you run **variants**, show them, let the user pick, then run **finish**.
The models are generative: they redraw detail, and **faces are what users judge hardest**. A tiny
difference in a familiar face reads as "that's not them". You cannot judge resemblance; the user
can. Show faces side by side, describe the differences honestly, and let them choose.

**Every time you work with variants, the user gets a comparison page:** `compare.html` in the
folder that holds them, opened from disk, flipping between the original and each variant at the
same size and place with the keys 1, 2, 3... `variants`, `mix` and `finish` write it for you; for any
other set, run `compare` (step 5). Always give the user its local path, never an online page with
their photos.

All paths below are relative to this skill's folder (the repo). Run commands from it:

```bash
cd ~/.claude/skills/local-upscale
PY=.venv/Scripts/python.exe
$PY scripts/upscale.py status                     # local-image, engine, weights, GPU
```

**`variants` and `finish` open their own visible window** ("local-upscale variants photo.jpg -
close this window to stop it") and run there; your command waits, echoes the window's lines (Qwen
edits, variants, SeedVR2 phases and tiles) and returns the job's exit code. Tell the user the window
is theirs: **closing it stops the job and frees the GPU** (for `variants` it also stops local-image's
server). To stop a job yourself, run `$PY scripts/upscale.py stop` (what stop.cmd runs); don't kill
processes by hand. Only one of these jobs runs at a time: if one refuses with "already running",
wait for it or ask the user before stopping it. A failed job's window stays up a minute to show the
error; the full log is in `logs/`.

## The loop

1. **Check the GPU** (`nvidia-smi`). local-image needs ~15.5 GB and SeedVR2 ~13 GB (3B ~10 GB), one at a time.
   If another program holds the GPU, ask the user before stopping anything (use the `local-gpu`
   skill if it's installed). Never kill processes you didn't start.
2. **Photo location.** Users drop photos in `input/`; any path works. Results go to `output/<photo>/`.
3. **Variants** (~3 min for small photos, ~8 min at 2048 px):
   ```bash
   $PY scripts/upscale.py variants "input/photo.jpg"             # restore
   $PY scripts/upscale.py variants "input/photo.jpg" --colorize  # black-and-white photo
   $PY scripts/upscale.py variants "input/photo.jpg" --ref "input/other.jpg"   # identity reference(s)
   ```
   Run it in the background and tell the user it's running in its window. Then **Read**
   `output/<photo>/variants/contact_sheet.jpg` and `faces.jpg` yourself, and tell the user what
   differs between the variants: face shape, expression, colour, sharpness, any invented text. Give
   them the local path of `output/<photo>/compare.html` (written by every `variants`, `mix` and
   `finish`): opened from disk, it shows the original and every variant at the same size and place,
   switched with the keys 1, 2, 3... (Space or holding the photo shows the original, Z full size).
   Let them pick.
4. **Finish** the pick (~1-2 min):
   ```bash
   $PY scripts/upscale.py finish "input/photo.jpg" --variant 3          # final = original size, or 2K
   $PY scripts/upscale.py finish "input/photo.jpg" --variant 3 --full   # keep the full upscale size
   $PY scripts/upscale.py finish "input/photo.jpg" --variant 3 --model 3b   # another installed upscaler
   ```
   Before reporting, Read `output/<photo>/work/faces_final_vN_<model>.jpg` (the finish log prints
   its path): close-ups of each face in the original, the picked variant and the final, all at the
   final size. Tell the user if the upscale changed a face. Deliver the paths:
   `<photo>_2_final_vN_<model>_*.png` is the result, `_3_..._sharpened` a crisper copy, and `_4_upscaled_*`
   the full SeedVR2 output. `compare.html` now lists the finals too.
5. **Any other set of images** you make for a photo (an extra edit in a folder of its own, say):
   `$PY scripts/upscale.py compare "output/photo/that_folder"` writes a `compare.html` there; a folder
   without an original of its own borrows the photo's from the folder above. Run it after every
   change to that set, and give the user its path. (local-image's `generate.py --out` also writes
   one into its folder, with the edited image first.)

## When faces aren't right

- **Each person looks best in a different variant:** mix them.
  ```bash
  $PY scripts/upscale.py mix "input/photo.jpg" --base 3              # lists faces, writes variants/faces_numbered.jpg
  $PY scripts/upscale.py mix "input/photo.jpg" --base 3 --face 1=2   # face 1 from variant 2 into variant 3
  ```
  **Read `faces_numbered.jpg` before choosing numbers.** Faces are numbered left to right, and the
  detector sometimes also boxes a hand or a printed face. The result is the next variant number;
  finish it.
- **Tiny faces** (a few dozen pixels): the model has to invent them, so resemblance is luck. Ask whether
  there's **another photo of the same people from the same day or period**, and re-run variants with
  `--ref`. This works far better than anything else tried. References from many years apart
  change the face instead; avoid them.
- **More choice:** run `variants` again; new variants get the next numbers, and the overviews show
  all that remain.
- If a photo is too small to give any resemblance (a face of ~60 px), say so plainly instead of
  iterating forever.

## Prints and scans

Photo of a print on a table, or a scan with the lid showing: crop first, then work on the result.
```bash
$PY scripts/upscale.py crop "input/scan.jpg"            # -> input/scan_cropped.jpg
$PY scripts/upscale.py crop "input/scan.jpg" --inset 0.03 --corners x1,y1,x2,y2,x3,y3,x4,y4
```
Always Read `output/scan/work/crop_check.jpg` (the detected outline). If it's wrong, pass `--corners`.

## Don'ts (learned the hard way)

- Don't upscale far beyond 2-4x ("16K"): it only magnifies noise. 2x then downscale is the sweet spot.
- Don't use local-image's Enhance as a restoration step: it keeps colour casts and thins faces.
- Don't run a batch of 4 at 2048 px in one Qwen call (VRAM overflow); the tool already splits it.
- Don't edit the user's originals. Everything goes to `output/`.
- Users often delete rejected variants or finals from the folders. That's expected; work with
  what's there.

## Licence note

Qwen-Image (step 1) is under a research/evaluation-only licence. If the user plans commercial use
of the results, mention it once.
