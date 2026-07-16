<div align="center">

<img src="./assets/logo.svg" alt="removed_bg logo" width="120" height="120" />

# removed_bg.py

**Programmatic background removal for images — two methods, one clean script.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Pillow](https://img.shields.io/badge/Pillow-10%2B-11557C?style=flat-square)](https://python-pillow.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![rembg](https://img.shields.io/badge/rembg-2.x-F97316?style=flat-square)](https://github.com/danielgatis/rembg)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Guide](https://img.shields.io/badge/Guide-Website-6366F1?style=flat-square&logo=github)](https://biraj2004.github.io/removed_bg/)

</div>

---

## Table of contents

- [Overview](#overview)
- [Guide website](#guide-website)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Input modes](#input-modes)
  - [All options](#all-options)
- [How it works](#how-it-works)
- [Output: quality report](#output-quality-report)
- [Examples](#examples)
- [Choosing the right tolerance](#choosing-the-right-tolerance-colour-key)
- [Tech stack](#tech-stack)
- [Limitations](#limitations)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`removed_bg.py` is a single-file Python CLI tool that strips the background from any image and saves a clean, transparent RGBA PNG. No API key, no internet connection after setup, no external service — everything runs locally.

---

## Guide website

Full documentation, usage examples, and interactive demos are available at:

**<https://biraj2004.github.io/removed_bg/>**

| | Method 1 — AI | Method 2 — Colour-key |
|---|---|---|
| **Best for** | Any image — photos, logos, complex scenes | Logos with a solid, uniform background |
| **Quality** | Professional (handles hair, fur, glass) | Good for flat-background images |
| **Speed** | Slower (model inference) | Fast (pure NumPy) |
| **Offline** | After first model download | Fully offline |
| **Extra deps** | `rembg` + `onnxruntime` | `Pillow` + `numpy` only |

---

## Installation

```bash
git clone https://github.com/Biraj2004/removed_bg
cd removed_bg

# Core dependencies (always needed)
pip install Pillow numpy

# AI method — CPU
pip install rembg onnxruntime

# AI method — NVIDIA GPU (faster inference)
pip install rembg onnxruntime-gpu
```

> **Python 3.8+** required. Tested on 3.9, 3.10, 3.11, and 3.12.

### GPU Acceleration Requirements (CUDA)
To use `--gpu` mode with `onnxruntime-gpu`, your machine must have:
1. A compatible NVIDIA GPU with installed drivers.
2. NVIDIA CUDA Toolkit (version 12.x matches `onnxruntime-gpu` 1.23+).
3. The CUDA `bin` path (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` on Windows) added to your system environment `Path` variable.
*(If CUDA DLLs are missing or incorrectly configured, the script will catch it and display setup instructions.)*

**Optional — pin versions for reproducible environments:**

```
# requirements.txt
Pillow>=10.0.0
numpy>=1.21.0
rembg>=2.0.50
onnxruntime>=1.16.0   # swap for onnxruntime-gpu on CUDA machines
```

```bash
pip install -r requirements.txt
```

---

## Quick start

```bash
# Single image — AI method (output auto-named photo_nobg.png)
python removed_bg.py photo.jpg

# Single image — colour-key, auto-detect background color (looks at corners)
python removed_bg.py signature.png --method colorkey

# Single image — colour-key, specify explicit background color (white)
python removed_bg.py logo.png logo_nobg.png --method colorkey --bg-color 255,255,255

# Batch — entire folder
python removed_bg.py images/ --batch --method ai --out-dir ./output/

# Batch — glob pattern
python removed_bg.py "images/*.png" --batch --method ai --out-dir ./output/
```

---

## Usage

```
python removed_bg.py <input> [output] [options]
python removed_bg.py <input> [input ...] --batch [options]
```

### Input modes

The script accepts three kinds of input, which can be combined freely in `--batch` mode.

**Single file**

Pass one file path. The output path is optional and defaults to `<stem>_nobg.png` in the same folder.

```bash
python removed_bg.py logo.png                   # → logo_nobg.png (same folder)
python removed_bg.py logo.png out/clean.png     # → explicit output path
```

**Glob pattern** (`--batch` required)

Always quote the pattern so the shell does not expand it — the script handles expansion itself, which also works correctly on Windows.

```bash
python removed_bg.py "images/*.png" --batch --out-dir ./nobg/
python removed_bg.py "icons/*.png" "banners/*.jpg" --batch --out-dir ./nobg/
```

**Directory** (`--batch` required)

Pass a folder path. The script scans for all supported image files (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) inside it, sorted alphabetically. Subdirectories are not traversed.

```bash
python removed_bg.py images/ --batch --out-dir ./nobg/
```

> If no supported images are found in the folder, the script prints a warning and exits cleanly.

### Output Naming Rules

The name and location of the saved output files are determined by the mode and arguments you supply:

| Mode | Command Example | Output Location & Naming |
|---|---|---|
| **Single File (Explicit)** | `python removed_bg.py in.jpg custom.png` | Saves exactly as `custom.png` at the specified path. |
| **Single File (Auto-derived)** | `python removed_bg.py in.jpg` | Appends `_nobg.png` to the file name and saves it in the original folder (`in_nobg.png`). |
| **Batch (with `--out-dir`)** | `python removed_bg.py images/ --batch --out-dir ./nobg/` | Saves all outputs as `<stem>_nobg.png` inside the target directory (e.g. `nobg/image_nobg.png`). |
| **Batch (without `--out-dir`)** | `python removed_bg.py images/ --batch` | Saves all outputs as `<stem>_nobg.png` in their respective original parent folders. |

### All options

| Flag | Default | Description |
|---|---|---|
| `output` | `<stem>_nobg.png` | Output PNG path (single-file mode only). |
| `--method` | `ai` | `ai` — deep learning. `colorkey` — solid colour removal. |
| `--model` | `isnet-general-use` | AI model to use (ai method only). Examples: `isnet-general-use` (default, high accuracy), `u2net` (standard), `u2netp` (fast/lightweight), `u2net_human_seg`, `silueta`, `sam`. |
| `--gpu` | — | Force NVIDIA GPU execution (`CUDAExecutionProvider`). Errors out if CUDA is unavailable. |
| `--bg-color R,G,B` | `auto` | Background colour to remove. If `auto` (default), samples the 4 corner pixels of the image and averages them. *(colorkey only)* |
| `--tolerance N` | `30` | RGB distance threshold. Raise to `50–80` for JPEG artefacts. *(colorkey only)* |
| `--feather N` | `3` | Gaussian blur radius on the alpha mask (0–10). *(colorkey only)* |
| `--batch` | — | Enable batch mode. Required for multiple files, globs, or a directory. |
| `--out-dir DIR` | — | Output directory for batch mode. Created automatically if absent. |

---

## How it works

### Method 1 — AI (U2-Net via rembg)

[rembg](https://github.com/danielgatis/rembg) runs the **U2-Net** deep learning model trained for salient object detection and alpha matting. It predicts a per-pixel foreground probability map — the same approach used by commercial tools like remove.bg.

```
Input image  →  U2-Net inference  →  Alpha matte  →  RGBA PNG
```

The `u2net.onnx` model (~170 MB) is downloaded on first run and cached at `~/.u2net/`. All subsequent runs are fully offline.

### Method 2 — Colour-key

A fast, dependency-light approach for images with a known uniform background:

```
1. Convert to RGBA float32
2. Per-pixel Euclidean distance from bg_color:
       dist = sqrt((R−Br)² + (G−Bg)² + (B−Bb)²)
3. Build alpha mask with feathered edge:
       dist < tolerance        →  0.0  (transparent)
       tolerance ≤ dist < 2×t  →  linear ramp
       dist ≥ 2×tolerance      →  1.0  (opaque)
4. Gaussian blur (radius = --feather) for sub-pixel anti-aliasing
5. Multiply mask × original alpha  →  write RGBA PNG
```

---

## Output: quality report

After every run the script prints an alpha-channel summary:

```
── Quality report ───────────────────────────────────────────────
   Size          :   1790 × 1790 px
   Transparent   :  1,717,673  (53.6%)
   Opaque        :  1,291,422  (40.3%)
   Semi-trans    :    195,005  ( 6.1%)  ← edge matting
   Corner alphas : [0, 0, 0, 0]        ← should all be ≈ 0 if BG removed
─────────────────────────────────────────────────────────────────
```

**Semi-transparent pixels** represent edge matting — a healthy range is **3–10%** of total pixels. Too low means hard jagged edges; too high may mean the background colour is bleeding into the subject.

---

## Examples

### AI — product photo

```bash
python removed_bg.py product.jpg product_nobg.png
```

### Colour-key — logo on black background

```bash
python removed_bg.py logo.png logo_nobg.png \
  --method colorkey \
  --bg-color 0,0,0 \
  --tolerance 40 \
  --feather 3
```

### Colour-key — JPEG with heavy compression artefacts

```bash
python removed_bg.py banner.jpg banner_nobg.png \
  --method colorkey \
  --bg-color 30,27,75 \
  --tolerance 65
```

### Batch — AI on an entire folder

```bash
python removed_bg.py photos/ --batch --method ai --out-dir ./transparent/
```

### Batch — colour-key on a glob pattern

```bash
python removed_bg.py "assets/*.png" --batch \
  --method colorkey \
  --bg-color 255,255,255 \
  --tolerance 20 \
  --out-dir ./nobg/
```

### Batch — multiple glob patterns in one call

```bash
python removed_bg.py "src/icons/*.png" "src/banners/*.jpg" \
  --batch --method ai --out-dir ./nobg/
```

---

## Tips for 99% Accuracy (Getting the Best Cutout)

To achieve perfect 99% background removal accuracy, choose the right method, model, and parameters based on the type of image:

### A. For Handwriting, Signatures, Logos, and Thin Line-Art
Salient object detection models (`u2net`/`isnet`) are trained to look for large foreground subjects. They will often strip away isolated details like **dots on `i` and `j`** or thin signature lines.

*   **Solution**: Use `--method colorkey` with the exact background color.
*   **Step 1**: Find the exact background color of the image corners (e.g. by using Python or a color picker):
    ```bash
    python -c "from PIL import Image; img = Image.open('your_image.jpg'); print('Corner color:', img.getpixel((0,0)))"
    ```
*   **Step 2**: Run the color-key method using that exact color and a custom tolerance (typically `40–60` for JPEG compression):
    ```bash
    python removed_bg.py signature.jpg clean.png --method colorkey --bg-color 235,235,235 --tolerance 50 --feather 2
    ```

### B. For Complex Photos, People, and Products
For general photos, the AI method (`--method ai`) is best. You can select the specific model that fits your subject:

*   **For Ultra-High Accuracy (e.g., hair, detailed silhouettes)**: Use the modern **`isnet-general-use`** model. It reduces semi-transparent edge artifacts by over 60% compared to the default model:
    ```bash
    python removed_bg.py photo.jpg clean.png --method ai --model isnet-general-use
    ```
*   **For Human Portraits / People**: Use **`u2net_human_seg`** for optimized skin/hair boundaries.
*   **For Clothes**: Use **`u2net_cloth_seg`**.
*   **For Fast/Low-Resource Environments**: Use **`u2netp`** (lightweight U2-Net) or **`silueta`** (43MB model).

### C. Choosing the Right Tolerance (Colour-key)

| Image type | Recommended `--tolerance` |
|---|---|
| Clean PNG / lossless logo | `20–35` |
| JPEG with minor compression | `40–55` |
| JPEG with heavy compression | `55–80` |
| Gradient or textured background | ❌ Use `--method ai` instead |

*   If the output still shows a dark or light fringe, raise `--tolerance` by 10 and retry.
*   To make edges softer and reduce jagged borders, use `--feather` (values between `1` and `4` are ideal).

---

## Tech stack

| Component | Purpose |
|---|---|
| **Python 3.8+** | Runtime |
| **[Pillow](https://python-pillow.org)** | Image I/O, Gaussian blur, RGBA operations |
| **[NumPy](https://numpy.org)** | Vectorised per-pixel distance and mask computation |
| **[rembg](https://github.com/danielgatis/rembg)** | U2-Net model wrapper (AI method) |
| **[onnxruntime](https://onnxruntime.ai)** | ONNX inference engine — CPU and CUDA |

> All processing is local. No data is sent to any server. No API key is required.

---

## Limitations

- **Colour-key** only works on images with a **single, uniform** background colour. Avoid using it on photos, gradients, or JPEG-compressed images — use `--method ai` instead.
- **AI method** needs the `u2net.onnx` model cached at `~/.u2net/`. In restricted network environments, pre-download it and place it at that path manually.
- **Directory scanning is non-recursive** — only files directly inside the given folder are processed; subdirectories are ignored.
- **Very fine detail** (individual hairs, wispy smoke, transparent glass) may not be perfectly matted even with the AI method and may benefit from manual touch-up.

---

## Security

To report a security vulnerability, please follow the process described in [SECURITY.md](SECURITY.md) rather than opening a public issue. The file covers supported versions, the private reporting channel, and the coordinated-disclosure policy.

---

## Contributing

Bug reports and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with Claude by **[Biraj](https://github.com/Biraj2004)**

</div>
