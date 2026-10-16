# Model and third-party licensing

**This repository contains no model weights and no third-party code.** `install.ps1` clones the
SeedVR2 CLI and downloads the weights at install time, pinned by commit and SHA-256 (`models.json`).
The code in this repo is MIT ([LICENSE](LICENSE)). What it downloads or calls carries its own
licence:

| Component | Licence | How it's used |
|---|---|---|
| **Qwen-Image-2.1** (through [local-image](https://github.com/vl3c/local-image), a separate install) | **Qwen Research License: research or evaluation purposes only** | Step 1: clean-up / colorize edits |
| SeedVR2 CLI ([numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)) | Apache-2.0 | Cloned into `engine\SeedVR2`, run as a separate process |
| SeedVR2 7B / 7B-sharp / 3B weights, whichever you install (ByteDance Seed; FP8 repacks by numz / AInVFX on Hugging Face) | Apache-2.0 | Step 3: upscaling |
| YuNet face detector ([OpenCV Zoo](https://github.com/opencv/opencv_zoo)) | MIT | Face close-ups and face positions for `mix` |
| PyTorch, OpenCV, NumPy and the SeedVR2 Python dependencies | Their own permissive or weak-copyleft licences (BSD, Apache-2.0, MIT, MPL-2.0, the Matplotlib licence); the CUDA libraries inside the PyTorch wheels are under NVIDIA's CUDA EULA | Installed into `.venv` from PyPI and the PyTorch index |

## What the Qwen licence means for your results

Every photo processed by this tool passes through Qwen-Image. Under the Qwen Research License
the permitted scope is **research or evaluation**. Restoring your own family photos for yourself is
the kind of use nobody is likely to object to, but it is not clearly inside the grant. Anything
commercial (paid restoration work, selling or publishing the results in a product or monetised
content) needs a separate commercial licence from Qwen. local-image's own `LICENSE-MODELS.md` quotes
the relevant clauses.

SeedVR2's step on its own is Apache-2.0, so a future version could offer a fully permissive path
(SeedVR2 restoration without the Qwen step) at some cost in quality.
