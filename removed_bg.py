"""
removed_bg.py — Image Background Removal Utility
================================================
Two battle-tested methods:

  METHOD 1 — AI / rembg (RECOMMENDED for general images)
      Deep learning via U2-Net/IS-Net. Handles photos, logos, complex scenes,
      hair, fur, glass, semi-transparent edges. Downloads the model on first
      run (cached at ~/.u2net/).

      Select a specific model (e.g. high-accuracy isnet-general-use):
      python removed_bg.py photo.jpg out.png --method ai --model isnet-general-use

      Run with forced NVIDIA GPU (CUDA) acceleration:
      python removed_bg.py photo.jpg out.png --method ai --gpu

  METHOD 2 — Colour-key (fast, preserves dots/fine details for uniform backgrounds)
      Pure NumPy + Pillow. Works perfectly on handwriting signatures, logos with uniform
      backgrounds (black, white, or any specific RGB). Uses Euclidean
      distance in RGB space with a feathered alpha ramp and Gaussian
      blur for smooth edges.

      python removed_bg.py signature.png out.png --method colorkey --bg-color 235,235,235 --tolerance 50 --feather 2

  Single file (output auto-derived as <stem>_nobg.png):
      python removed_bg.py logo.png --method colorkey

  Batch processing — glob pattern:
      python removed_bg.py "images/*.png" --batch --method ai --out-dir ./nobg/

  Batch processing — entire directory:
      python removed_bg.py images/ --batch --method ai --out-dir ./nobg/

  Requirements:
      pip install Pillow numpy
      pip install rembg onnxruntime          # for --method ai (CPU)
      pip install rembg onnxruntime-gpu      # for --method ai (CUDA GPU)

Author : Biraj  ·  github.com/Biraj2004
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Module-level quiet flag — set True by --quiet
_QUIET: bool = False


# Supported image extensions for directory scanning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — AI-based removal (rembg / U2-Net)
# ─────────────────────────────────────────────────────────────────────────────

def remove_bg_ai(
    input_path: str,
    output_path: str,
    model_name: str = "u2net",
    force_gpu: bool = False,
) -> Image.Image:
    """
    Remove background using deep learning model via rembg.

    Produces a clean RGBA PNG with proper alpha matting on all edges,
    including semi-transparent regions, hair, and complex silhouettes.
    Downloads the ONNX model on first run and caches it.

    Args:
        input_path:  Path to the input image (PNG, JPG, WEBP, etc.).
        output_path: Path to write the output RGBA PNG.
        model_name:  The name of the rembg model to use (default: u2net).
        force_gpu:   Whether to force using the GPU (CUDAExecutionProvider).

    Returns:
        PIL Image (RGBA) of the result.
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        _die(
            "rembg is not installed.\n"
            "  Run:  pip install rembg onnxruntime\n"
            "  GPU:  pip install rembg onnxruntime-gpu"
        )

    _log(f"[AI] Input  : {input_path}")
    img = Image.open(input_path)

    # Configure execution providers for ONNX Runtime
    providers = None
    if force_gpu:
        providers = ["CUDAExecutionProvider"]
        _log("[AI] Forcing NVIDIA GPU (CUDAExecutionProvider)...")
    else:
        # Check if CUDA is available in onnxruntime
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                _log("[AI] NVIDIA GPU (CUDAExecutionProvider) is available.")
            else:
                _log("[AI] CUDAExecutionProvider not found in ONNX Runtime. Falling back to CPU.")
        except Exception:
            pass

    _log(f"[AI] Initializing session with model '{model_name}'...")
    try:
        session = new_session(model_name, providers=providers)
        # Verify if CUDA was actually loaded when force_gpu is True
        if force_gpu and "CUDAExecutionProvider" not in session.inner_session.get_providers():
            _die(
                "NVIDIA GPU (CUDAExecutionProvider) was requested, but ONNX Runtime is running on CPU.\n"
                "Error details: ONNX Runtime failed to load CUDA DLLs (e.g. cublasLt64_12.dll is missing).\n"
                "To resolve this, please install NVIDIA CUDA Toolkit 12.x and add it to your Windows PATH:\n"
                "  1. Download: https://developer.nvidia.com/cuda-downloads\n"
                "  2. Add 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin' to your System Env Path\n"
                "  3. Restart your terminal / editor and try again."
            )
    except Exception as e:
        if force_gpu:
            _die(
                f"Failed to initialize CUDA session: {e}\n"
                "Please check your CUDA/cuDNN installation."
            )
        else:
            _log(f"[WARN] Session initialization failed: {e}. Retrying with defaults...")
            session = new_session(model_name)

    _log(f"[AI] Running model inference (downloads '{model_name}' on first run if needed) ...")
    result: Image.Image = remove(img, session=session)

    _ensure_dir(output_path)
    result.save(output_path)
    _log(f"[AI] Output : {output_path}")
    _quality_report(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — Colour-key removal with alpha matting
# ─────────────────────────────────────────────────────────────────────────────

def remove_bg_colorkey(
    input_path: str,
    output_path: str,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    tolerance: int = 30,
    edge_feather: int = 3,
) -> Image.Image:
    """
    Remove a solid-colour background with smooth edge alpha matting.

    Algorithm
    ---------
    1. Convert image to RGBA float32.
    2. Compute per-pixel Euclidean distance from bg_color in RGB space:
           dist = sqrt((R - Br)² + (G - Bg)² + (B - Bb)²)
    3. Build a [0, 1] alpha mask:
         - dist < tolerance          → 0.0  (fully transparent)
         - tolerance ≤ dist < 2×tol  → linear ramp  (feather zone)
         - dist ≥ 2×tolerance        → 1.0  (fully opaque)
    4. Apply Gaussian blur (radius = edge_feather) to the mask for
       sub-pixel anti-aliasing on edges.
    5. Multiply the mask against the original alpha channel and write back.

    Args:
        input_path:   Path to the input image.
        output_path:  Path to write the output RGBA PNG.
        bg_color:     (R, G, B) of the background colour to remove.
                      Default (0, 0, 0) = black.
        tolerance:    Pixels within this distance of bg_color become
                      transparent. Increase to 50–80 for JPEG-compressed
                      images with artefacts. Default 30.
        edge_feather: Gaussian blur radius applied to the alpha mask.
                      Higher values = softer, wider edge transition.
                      Default 3.

    Returns:
        PIL Image (RGBA) of the result.

    Limitations
    -----------
    Only reliable when the background is a SINGLE, UNIFORM colour.
    Fails on photos, gradients, and textured backgrounds — use AI method
    for those.
    """
    _log(f"[CK] Input  : {input_path}")
    _log(f"[CK] BG colour  = {bg_color}")
    _log(f"[CK] Tolerance  = {tolerance}")
    _log(f"[CK] Feather    = {edge_feather}")

    img = Image.open(input_path).convert("RGBA")
    data = np.array(img, dtype=np.float32)  # shape: (H, W, 4)

    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    br, bg_, bb = float(bg_color[0]), float(bg_color[1]), float(bg_color[2])

    # ── Euclidean distance in RGB space ──────────────────────────────────────
    dist = np.sqrt((r - br) ** 2 + (g - bg_) ** 2 + (b - bb) ** 2)

    # ── Alpha mask: 0 = transparent, 1 = opaque ──────────────────────────────
    feather_end = float(tolerance * 2)
    denom = max(feather_end - tolerance, 1.0)
    mask = np.clip((dist - tolerance) / denom, 0.0, 1.0)

    # ── Gaussian blur for anti-aliased edges ─────────────────────────────────
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    if edge_feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=edge_feather))
    mask_f = np.array(mask_img, dtype=np.float32) / 255.0

    # ── Apply mask — preserve any existing alpha ──────────────────────────────
    orig_alpha = data[:, :, 3] / 255.0
    new_alpha = (orig_alpha * mask_f * 255.0).clip(0, 255).astype(np.uint8)

    out = data.astype(np.uint8).copy()
    out[:, :, 3] = new_alpha
    result = Image.fromarray(out, "RGBA")

    _ensure_dir(output_path)
    result.save(output_path)
    _log(f"[CK] Output : {output_path}")
    _quality_report(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quality_report(img: Image.Image) -> None:
    """Print a brief alpha-channel quality summary."""
    data = np.array(img.convert("RGBA"))
    alpha = data[:, :, 3]
    total = alpha.size
    transparent = int(np.sum(alpha < 10))
    opaque = int(np.sum(alpha > 245))
    semi = total - transparent - opaque
    corners = [int(data[0, 0, 3]), int(data[0, -1, 3]),
               int(data[-1, 0, 3]), int(data[-1, -1, 3])]

    print()
    print("== Quality report ===============================================")
    print(f"   Size          : {img.size[0]} x {img.size[1]} px")
    print(f"   Transparent   : {transparent:>10,}  ({100*transparent/total:5.1f}%)")
    print(f"   Opaque        : {opaque:>10,}  ({100*opaque/total:5.1f}%)")
    print(f"   Semi-trans    : {semi:>10,}  ({100*semi/total:5.1f}%)  <- edge matting")
    print(f"   Corner alphas : {corners}  <- should all be ~ 0 if BG removed")
    print("=================================================================")
    print()


def _log(msg: str) -> None:
    if not _QUIET:
        print(msg, flush=True)


def _die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _ensure_dir(path: str) -> None:
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


def _output_name(input_path: str, out_dir: str | None) -> str:
    """Derive output path: <out_dir>/<stem>_nobg.png or <stem>_nobg.png."""
    stem = Path(input_path).stem
    name = f"{stem}_nobg.png"
    if out_dir:
        return str(Path(out_dir) / name)
    return str(Path(input_path).parent / name)


def _expand_paths(raw_inputs: list[str]) -> list[str]:
    """
    Resolve a mixed list of file paths, glob patterns, and directories
    into a flat list of image file paths.

    - Plain file path  → kept as-is (error surfaces at open time if missing)
    - Glob pattern     → expanded via glob.glob
    - Directory path   → all image files directly inside it (non-recursive)
                         sorted alphabetically; subdirectories are skipped
    """
    result: list[str] = []
    for raw in raw_inputs:
        p = Path(raw)
        if p.is_dir():
            # Expand directory to all supported image files (non-recursive)
            found = sorted(
                str(f) for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not found:
                _log(f"[WARN] No supported image files found in directory: {raw}")
            result.extend(found)
        else:
            expanded = glob.glob(raw)
            if expanded:
                result.extend(sorted(expanded))
            else:
                result.append(raw)  # keep as-is; error will surface at open time
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_color(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Color must be R,G,B  (e.g. 0,0,0 or 255,255,255).  Got: {s!r}"
        )
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"All components must be integers.  Got: {s!r}")
    for v in (r, g, b):
        if not (0 <= v <= 255):
            raise argparse.ArgumentTypeError(f"Each component must be 0–255.  Got: {v}")
    return (r, g, b)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="removed_bg.py",
        description="Remove the background from an image — AI or colour-key method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    p.add_argument(
        "input",
        nargs="+",
        help=(
            "Input image path(s). "
            "Accepts: a single file path, a glob pattern (e.g. 'images/*.png'), "
            "or a directory path (all supported images inside are processed). "
            "Use --batch when supplying multiple files or a directory."
        ),
    )
    p.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output PNG path. Omit when using --batch "
            "(names are derived automatically as <stem>_nobg.png). "
            "Also optional for single-file mode — defaults to <stem>_nobg.png."
        ),
    )
    p.add_argument(
        "--method",
        choices=["ai", "colorkey"],
        default="ai",
        help=(
            "Removal method.\n"
            "  ai       : U2-Net deep learning via rembg (default, works on any image).\n"
            "  colorkey : Fast solid-colour removal via NumPy + Pillow.\n"
        ),
    )
    p.add_argument(
        "--bg-color",
        default="0,0,0",
        type=_parse_color,
        metavar="R,G,B",
        help=(
            "Background colour to remove (colorkey only).  "
            "Default: 0,0,0 (black).  "
            "Example: --bg-color 255,255,255 for white."
        ),
    )
    p.add_argument(
        "--tolerance",
        type=int,
        default=30,
        metavar="N",
        help=(
            "Colour distance tolerance for colorkey (default 30). "
            "Pixels within this Euclidean distance of --bg-color become transparent. "
            "Raise to 50–80 for JPEG-compressed images with artefacts."
        ),
    )
    p.add_argument(
        "--feather",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Gaussian blur radius applied to the alpha mask (colorkey only). "
            "Controls edge softness. Default 3. Range 0–10."
        ),
    )
    p.add_argument(
        "--model",
        default="u2net",
        help=(
            "AI model to use (ai method only). "
            "Examples: u2net (default), u2netp (lightweight), isnet-general-use (high accuracy), "
            "u2net_human_seg, silueta, sam."
        ),
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Force using NVIDIA GPU (CUDAExecutionProvider) for AI model. Errors out if CUDA is unavailable.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Process multiple files. Input arguments may be explicit file paths, "
            "glob patterns (e.g. 'images/*.png'), or a directory path. "
            "Output filenames are auto-derived as <stem>_nobg.png. "
            "Use --out-dir to control where outputs are written."
        ),
    )
    p.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help="Directory for output files when using --batch. Created if absent.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all log output (errors are still printed to stderr).",
    )
    return p


def main() -> None:
    global _QUIET
    parser = _build_parser()
    args = parser.parse_args()
    _QUIET = args.quiet

    # If two positional arguments are provided without --batch, argparse groups
    # both into args.input due to nargs="+". We reconstruct the intended input and output.
    if not args.batch and len(args.input) == 2 and args.output is None:
        args.output = args.input[1]
        args.input = [args.input[0]]

    # ── Resolve input paths (files, globs, and directories) ──────────────────
    paths = _expand_paths(args.input)

    if not paths:
        _die("No input files found. Check your path, glob pattern, or directory.")

    if args.batch:
        if args.output is not None:
            parser.error(
                "--batch mode: do not supply a positional output path. "
                "Use --out-dir instead."
            )
        total = len(paths)
        for i, p in enumerate(paths, 1):
            _log(f"[{i}/{total}] {p}")
            out = _output_name(p, args.out_dir)
            _run(args, p, out)
    else:
        if len(paths) > 1:
            parser.error(
                f"Multiple inputs resolved ({len(paths)} files) but --batch was not set. "
                "Pass --batch to process multiple files or a directory."
            )
        input_path = paths[0]
        output_path = args.output or _output_name(input_path, args.out_dir)
        _run(args, input_path, output_path)


def _run(args: argparse.Namespace, input_path: str, output_path: str) -> None:
    if args.method == "ai":
        remove_bg_ai(input_path, output_path, model_name=args.model, force_gpu=args.gpu)
    else:
        remove_bg_colorkey(
            input_path,
            output_path,
            bg_color=args.bg_color,
            tolerance=args.tolerance,
            edge_feather=args.feather,
        )


if __name__ == "__main__":
    main()
