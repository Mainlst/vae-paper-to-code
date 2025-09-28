#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import List, Tuple

import numpy as np


def imread(path: str) -> np.ndarray:
    """Read an image as uint8 HxWxC (C in {1,3}).

    Uses matplotlib to avoid adding new deps. Drops alpha channel if present.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        arr = plt.imread(path)
        # matplotlib may return float in [0,1] or uint8
        if arr.dtype != np.uint8:
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {e}")

    # Ensure shape is HxWxC
    if arr.ndim == 2:
        arr = arr[..., None]  # grayscale -> HWC with C=1
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha
    return arr


def imsave(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(path)
        return
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt  # type: ignore
        if img.ndim == 2 or img.shape[2] == 1:
            plt.imsave(path, img.squeeze(-1), cmap="gray")
        else:
            plt.imsave(path, img)
    except Exception as e:
        raise RuntimeError(f"Failed to save image {path}: {e}")


def _resize(im: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if im.ndim == 2:
        mode = "L"
    elif im.shape[2] == 1:
        mode = "L"
        im = im.squeeze(-1)
    else:
        mode = "RGB" if im.shape[2] >= 3 else "L"
        if im.shape[2] > 3:
            im = im[:, :, :3]
    try:
        from PIL import Image  # type: ignore
        pil = Image.fromarray(im, mode=mode)
        pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
        arr = np.asarray(pil)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr.astype(np.uint8)
    except Exception:
        # Nearest-neighbor fallback using numpy (rough)
        y_idx = (np.linspace(0, im.shape[0] - 1, new_h)).astype(int)
        x_idx = (np.linspace(0, im.shape[1] - 1, new_w)).astype(int)
        if im.ndim == 2:
            out = im[y_idx][:, x_idx]
            return out[..., None].astype(np.uint8)
        else:
            out = im[y_idx][:, x_idx, :]
            return out.astype(np.uint8)


def tile_images(
    images: List[np.ndarray],
    cols: int,
    padding: int = 2,
    pad_color: int = 0,
    inner_scale: float = 1.0,
    frame_px: int = 0,
    frame_color: int = 255,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if not images:
        raise ValueError("No images provided to tile_images")

    # Normalize sizes by padding smaller images to match the max size
    h_max = max(im.shape[0] for im in images)
    w_max = max(im.shape[1] for im in images)
    c = images[0].shape[2] if images[0].ndim == 3 else 1

    normed: List[np.ndarray] = []
    for im in images:
        if im.ndim == 2:
            im = im[..., None]
        if im.shape[2] != c:
            # convert to common channel count
            if im.shape[2] == 1 and c == 3:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 3 and c == 1:
                im = im[:, :, :1]
            else:
                # fallback: force to 3 channels
                c = 3
                im = im[:, :, :3] if im.shape[2] >= 3 else np.repeat(im, 3, axis=2)
        pad_h = h_max - im.shape[0]
        pad_w = w_max - im.shape[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Unexpected negative padding; bug in size normalization")
        if pad_h or pad_w:
            pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
            im = np.pad(im, pad_spec, mode="constant", constant_values=pad_color)
        normed.append(im)

    n = len(normed)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    grid_h = rows * h_max + padding * (rows - 1)
    grid_w = cols * w_max + padding * (cols - 1)
    grid = np.full((grid_h, grid_w, c), pad_color, dtype=np.uint8)

    # compute resized target dimensions
    inner_scale = max(0.1, min(inner_scale, 1.0))
    target_h = max(1, int(round(h_max * inner_scale)))
    target_w = max(1, int(round(w_max * inner_scale)))

    for idx, im in enumerate(normed):
        r = idx // cols
        cc = idx % cols
        y = r * (h_max + padding)
        x = cc * (w_max + padding)

        # Resize image if needed
        im_resized = im
        if im.shape[0] != target_h or im.shape[1] != target_w:
            im_resized = _resize(im, target_w, target_h)

        # Compute frame placement centered in the cell
        frame_total_h = target_h + 2 * frame_px
        frame_total_w = target_w + 2 * frame_px
        fy0 = y + max(0, (h_max - frame_total_h) // 2)
        fx0 = x + max(0, (w_max - frame_total_w) // 2)
        fy1 = min(y + h_max, fy0 + frame_total_h)
        fx1 = min(x + w_max, fx0 + frame_total_w)

        # Draw white frame if requested
        if frame_px > 0:
            grid[fy0:fy1, fx0:fx1, :] = frame_color

        # Paste image inside frame
        iy0 = fy0 + frame_px
        ix0 = fx0 + frame_px
        iy1 = min(iy0 + im_resized.shape[0], y + h_max)
        ix1 = min(ix0 + im_resized.shape[1], x + w_max)
        ih = iy1 - iy0
        iw = ix1 - ix0
        grid[iy0:iy1, ix0:ix1, : im_resized.shape[2]] = im_resized[:ih, :iw, : im_resized.shape[2]]

    # Return grid and geometry (rows, cols, h_max, w_max)
    return grid, (rows, cols, h_max, w_max)


def natural_key(s: str):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a montage from sample images across epochs")
    ap.add_argument("--input-dir", required=True, help="Directory containing per-epoch PNGs")
    ap.add_argument("--output", required=True, help="Output PNG path for montage")
    ap.add_argument("--pattern", default="epoch_*.png", help="Glob pattern to select images")
    ap.add_argument("--cols", type=int, default=5, help="Number of columns in the montage grid")
    ap.add_argument("--max", dest="max_images", type=int, default=0, help="Max number of images (0=all)")
    ap.add_argument("--padding", type=int, default=2, help="Padding between tiles in pixels")
    ap.add_argument("--no-labels", action="store_true", help="Disable per-image epoch labels")
    ap.add_argument("--font-size", type=int, default=16, help="Label font size in pixels")
    ap.add_argument("--stride", type=int, default=1, help="Pick every Nth image (e.g., 2)")
    ap.add_argument("--inner-scale", type=float, default=1.0, help="Scale factor inside each cell (0-1]")
    ap.add_argument("--frame-px", type=int, default=0, help="White frame width in pixels around each tile")
    ap.add_argument("--pad-color", type=int, default=0, help="Background color (0-255)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)), key=natural_key)
    if not files:
        raise SystemExit(f"No images match {os.path.join(args.input_dir, args.pattern)}")

    # Apply stride and max
    if args.stride > 1:
        files = files[:: args.stride]
    if args.max_images and args.max_images > 0:
        files = files[: args.max_images]

    images = [imread(p) for p in files]
    grid, geom = tile_images(
        images,
        cols=args.cols,
        padding=args.padding,
        pad_color=args.pad_color,
        inner_scale=args.inner_scale,
        frame_px=args.frame_px,
        frame_color=255,
    )

    if args.no_labels:
        imsave(args.output, grid)
        print(f"Saved montage: {args.output} ({len(files)} images)")
        return

    # Prepare labels: extract epoch number from filename like epoch_0001.png → 1
    import re
    labels: List[str] = []
    for p in files:
        m = re.search(r"epoch_(\d+)", os.path.basename(p))
        if m:
            labels.append(str(int(m.group(1))))
        else:
            # fallback to basename without extension
            labels.append(os.path.splitext(os.path.basename(p))[0])

    # Draw labels using matplotlib to avoid hard dependency on Pillow fonts
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patheffects as pe  # type: ignore

        rows, cols, h_max, w_max = geom
        H, W = grid.shape[:2]
        # For grayscale HxWx1, drop channel for imshow
        show_grid = grid.squeeze(-1) if grid.ndim == 3 and grid.shape[2] == 1 else grid
        dpi = 100
        fig_w = W / dpi
        fig_h = H / dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(show_grid, cmap="gray" if (grid.ndim == 3 and grid.shape[2] == 1) else None)
        ax.set_axis_off()

        # Place text near top-left of each tile with readable background
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(labels):
                    break
                # Position label inside the frame area
                target_h = max(1, int(round(h_max * max(0.1, min(args.inner_scale, 1.0)))))
                target_w = max(1, int(round(w_max * max(0.1, min(args.inner_scale, 1.0)))))
                frame_total_h = target_h + 2 * args.frame_px
                frame_total_w = target_w + 2 * args.frame_px
                fy0 = r * (h_max + args.padding) + max(0, (h_max - frame_total_h) // 2)
                fx0 = c * (w_max + args.padding) + max(0, (w_max - frame_total_w) // 2)
                y = fy0 + 4  # a bit inside the frame
                x = fx0 + 4
                ax.text(
                    x,
                    y,
                    labels[idx],
                    color="white",
                    fontsize=args.font_size,
                    fontweight="bold",
                    va="top",
                    ha="left",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                    bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.2", edgecolor="none"),
                )
                idx += 1

        fig.savefig(args.output, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved montage with labels: {args.output} ({len(files)} images)")
    except Exception as e:
        # Fallback: save without labels if plotting fails
        imsave(args.output, grid)
        print(f"[warn] labeling failed ({e}); saved montage without labels: {args.output}")


if __name__ == "__main__":
    main()
