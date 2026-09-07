"""
Image Split — place vertical and horizontal crop lines, then export tiles.
Requires: pip install -r requirements.txt
Run: python app.py
"""
from __future__ import annotations

import math
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import AbstractSet, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageTk

try:
    from circle_extract import CIRCLE_EXTRACT_OK, extract_circle_rgba
except ImportError:  # pragma: no cover
    CIRCLE_EXTRACT_OK = False  # type: ignore[misc, assignment]

    def extract_circle_rgba(_: Image.Image) -> Optional[Image.Image]:  # type: ignore[no-redef, misc]
        return None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_OK = True
except ImportError:  # pragma: no cover
    DND_OK = False
    DND_FILES = None  # type: ignore[assignment]
    TkinterDnD = None  # type: ignore[assignment]


def load_image_file(path: str) -> Tuple[Optional[Image.Image], str]:
    """Open and normalize an image file for editing. Returns (image, err); err is "" on success."""
    try:
        im = Image.open(path)
        im.load()  # force decode now so a truncated/corrupt file raises here, not later
    except OSError as err:
        return None, str(err)
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    elif im.mode not in ("RGB", "RGBA", "L", "LA", "1"):
        im = im.convert("RGB")
    return im, ""


def compute_tiles(
    image: Image.Image, vertical: List[int], horizontal: List[int]
) -> List[Tuple[Image.Image, int, int]]:
    """Return cropped regions; vertical = x positions, horizontal = y positions (sorted, inside image)."""
    w, h = image.size
    xs = sorted([0] + [x for x in vertical if 0 < x < w] + [w])
    ys = sorted([0] + [y for y in horizontal if 0 < y < h] + [h])
    tiles: List[Tuple[Image.Image, int, int]] = []
    for r, (y0, y1) in enumerate(zip(ys, ys[1:])):
        for c, (x0, x1) in enumerate(zip(xs, xs[1:])):
            if x1 <= x0 or y1 <= y0:
                continue
            box = (x0, y0, x1, y1)
            tiles.append((image.crop(box), r, c))
    return tiles


@dataclass
class SaveExportOptions:
    """User-controlled encoding settings for lossy and PNG compression."""

    jpeg_quality: int = 92  # 1–100
    webp_quality: int = 90  # 0–100
    webp_method: int = 6  # 0=fast / 6=slow-smaller; Pillow WebP
    png_compress: int = 6  # 0–9


# Export: "png" | "jpeg" | "webp" (lowercase)
def _tile_for_jpeg(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode in ("RGB", "L", "1"):
        return im.convert("RGB")
    return im.convert("RGB")


def save_tile_to_path(
    path: str,
    tile: Image.Image,
    file_format: str,
    options: Optional[SaveExportOptions] = None,
) -> None:
    """file_format: png | jpeg | webp. Optional encoding options (defaults are reasonable)."""
    o = options or SaveExportOptions()
    f = (file_format or "png").lower()
    if f in ("jpg", "jpeg", "jpe"):
        q = max(1, min(100, int(o.jpeg_quality)))
        _tile_for_jpeg(tile).save(
            path, format="JPEG", quality=q, optimize=True, subsampling=0
        )
    elif f == "webp":
        q = max(0, min(100, int(o.webp_quality)))
        m = max(0, min(6, int(o.webp_method)))
        tile.save(path, format="WEBP", quality=q, method=m)
    else:
        t = tile
        if t.mode == "P" and "transparency" in t.info:
            t = t.convert("RGBA")
        cl = max(0, min(9, int(o.png_compress)))
        t.save(path, format="PNG", compress_level=cl, optimize=True)


def _export_ext_for_format(file_format: str) -> str:
    f = (file_format or "png").lower()
    if f in ("jpg", "jpeg", "jpe"):
        return "jpg"
    if f == "webp":
        return "webp"
    return "png"


def _effective_format_for_saving(
    file_format: str, transparent_circle_cutout: bool
) -> str:
    """Alpha-friendly formats; JPEG is upgraded to PNG when a circle is applied."""
    if not transparent_circle_cutout:
        return file_format
    f = (file_format or "png").lower()
    if f in ("jpg", "jpeg", "jpe"):
        return "png"
    return file_format


def export_tiles(
    image: Image.Image,
    vertical: List[int],
    horizontal: List[int],
    out_dir: str,
    basename: str,
    file_format: str = "png",
    transparent_circle_cutout: bool = False,
    save_options: Optional[SaveExportOptions] = None,
    ignored: Optional[AbstractSet[Tuple[int, int]]] = None,
) -> Tuple[int, List[str]]:
    """Returns (saved_count, errors); a per-tile save failure is collected, not raised."""
    eff = _effective_format_for_saving(file_format, transparent_circle_cutout)
    ext = _export_ext_for_format(eff)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(basename or "image")[0]
    skip: AbstractSet[Tuple[int, int]] = ignored or frozenset()
    count = 0
    errors: List[str] = []
    for tile, r, c in compute_tiles(image, vertical, horizontal):
        if (r, c) in skip:
            continue
        out = tile
        if transparent_circle_cutout and CIRCLE_EXTRACT_OK:
            t2 = extract_circle_rgba(tile)
            if t2 is not None:
                out = t2
        p = os.path.join(out_dir, f"{stem}_r{r}_c{c}.{ext}")
        try:
            save_tile_to_path(p, out, eff, save_options)
            count += 1
        except OSError as err:
            errors.append(f"{os.path.basename(p)}: {err}")
    return count, errors


def _refine_runs_1d(mask: "np.ndarray", min_len: int) -> List[Tuple[int, int]]:
    """Consecutive True regions; merge if separated by 1–2 Falses; require len >= min_len."""
    if np is None:  # pragma: no cover
        return []
    a = np.asarray(mask, dtype=bool)
    n = int(a.size)
    i = 0
    out: List[Tuple[int, int]] = []
    while i < n:
        if not a[i]:
            i += 1
            continue
        j = i
        while j < n and a[j]:
            j += 1
        if j - i >= 1:
            if out and i - out[-1][1] <= 2:
                lo, _ = out.pop()
                out.append((lo, j))
            else:
                out.append((i, j))
        i = j
    return [r for r in out if (r[1] - r[0]) >= min_len]


def _expected_gutter_line_count(n_r_or_c: int) -> int:
    """Gutter *edge* line positions for a grid with n_r_or_c content rows or columns (≥1)."""
    return 2 * max(0, n_r_or_c - 1)


def _montage_cell_bands(
    w: int,
    h: int,
    nrows: int,
    ncols: int,
    vx: List[int],
    hy: List[int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], bool]:
    """
    For a table with white gutters: vx/hy are sorted x/y of gutter band edges
    (two per inter-column/row gap). Produces (xb, yb) content rectangles, row-major.
    """
    if ncols < 1 or nrows < 1 or w < 1 or h < 1:
        return [], [], False
    ex, ey = _expected_gutter_line_count(ncols), _expected_gutter_line_count(nrows)
    if len(vx) != ex or len(hy) != ey:
        return [], [], False
    xb: List[Tuple[int, int]] = []
    yb: List[Tuple[int, int]] = []
    if ncols == 1:
        xb = [(0, w)]
    else:
        if not (0 < vx[0] and vx[-1] < w):
            return [], [], False
        for i in range(0, 2 * (ncols - 1) - 1, 2):
            if not (vx[i] < vx[i + 1]):
                return [], [], False
        for j in range(ncols):
            if j == 0:
                xb.append((0, vx[0]))
            elif j < ncols - 1:
                xb.append((vx[2 * j - 1], vx[2 * j]))
            else:
                xb.append((vx[2 * ncols - 3], w))
    if nrows == 1:
        yb = [(0, h)]
    else:
        if not (0 < hy[0] and hy[-1] < h):
            return [], [], False
        for i in range(0, 2 * (nrows - 1) - 1, 2):
            if not (hy[i] < hy[i + 1]):
                return [], [], False
        for r in range(nrows):
            if r == 0:
                yb.append((0, hy[0]))
            elif r < nrows - 1:
                yb.append((hy[2 * r - 1], hy[2 * r]))
            else:
                yb.append((hy[2 * nrows - 3], h))
    for x0, x1 in xb:
        if not (0 <= x0 < x1 <= w):
            return [], [], False
    for y0, y1 in yb:
        if not (0 <= y0 < y1 <= h):
            return [], [], False
    return xb, yb, True


def get_tile_at_pixel(
    im: Image.Image,
    v: List[int],
    h: List[int],
    ix: int,
    iy: int,
    montage_table: bool,
    nrows: int = 0,
    ncols: int = 0,
) -> Optional[Tuple[Image.Image, int, int]]:
    w, h_ = im.size
    if not (0 <= ix < w and 0 <= iy < h_):
        return None
    exb: List[Tuple[int, int]] = []
    eyb: List[Tuple[int, int]] = []
    if (
        montage_table
        and nrows >= 1
        and ncols >= 1
        and len(v) == _expected_gutter_line_count(ncols)
        and len(h) == _expected_gutter_line_count(nrows)
    ):
        exb, eyb, ok = _montage_cell_bands(
            w, h_, nrows, ncols, sorted(v), sorted(h)
        )
        if not ok or not exb or not eyb:
            return None
    if not exb:
        if not v and not h:
            return (im, 0, 0)
        xs = sorted([0] + [x for x in v if 0 < x < w] + [w])
        ys = sorted([0] + [y for y in h if 0 < y < h_] + [h_])
        exb = list(zip(xs, xs[1:]))
        eyb = list(zip(ys, ys[1:]))
    for r, (ys0, ys1) in enumerate(eyb):
        for c, (xs0, xs1) in enumerate(exb):
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            if xs0 <= ix < xs1 and ys0 <= iy < ys1:
                return (im.crop((xs0, ys0, xs1, ys1)), r, c)
    return None


def list_tile_rects(
    im: Image.Image,
    v: List[int],
    h: List[int],
    montage_table: bool,
    nrows: int,
    ncols: int,
) -> List[Tuple[Tuple[int, int, int, int], int, int]]:
    """Each ((x0,y0,x1,y1) image px), row, col) — same cell partition as get_tile_at_pixel."""
    w, h_ = im.size
    exb: List[Tuple[int, int]] = []
    eyb: List[Tuple[int, int]] = []
    if (
        montage_table
        and nrows >= 1
        and ncols >= 1
        and len(v) == _expected_gutter_line_count(ncols)
        and len(h) == _expected_gutter_line_count(nrows)
    ):
        exb, eyb, ok = _montage_cell_bands(w, h_, nrows, ncols, sorted(v), sorted(h))
        if not ok or not exb or not eyb:
            exb, eyb = [], []
    if not exb:
        if not v and not h:
            return [((0, 0, w, h_), 0, 0)]
        xs = sorted([0] + [x for x in v if 0 < x < w] + [w])
        ys = sorted([0] + [y for y in h if 0 < y < h_] + [h_])
        exb = list(zip(xs, xs[1:]))
        eyb = list(zip(ys, ys[1:]))
    out: List[Tuple[Tuple[int, int, int, int], int, int]] = []
    for r, (ys0, ys1) in enumerate(eyb):
        for c, (xs0, xs1) in enumerate(exb):
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            out.append(((xs0, ys0, xs1, ys1), r, c))
    return out


def _scale_gutter_pair(
    p: Tuple[int, int], span_from: int, span_to: int
) -> Tuple[int, int]:
    f = span_to / max(1, span_from)
    lo, hi = p[0], p[1]
    a0, a1 = int(round(lo * f)), int(round(hi * f))
    if a1 - a0 < 1 and hi > lo:
        a1 = a0 + 1
    return a0, a1


def _pick_k_gutter_runs(
    runs: List[Tuple[int, int]], span: int, k: int
) -> List[Tuple[int, int]]:
    if k <= 0:
        return []
    if not runs:
        return []
    sruns = sorted(runs, key=lambda r: r[0])
    if k == 2 and len(sruns) == 1:
        lo, hi = sruns[0]
        if hi - lo < 8:
            return []
        t = (lo * 2 + hi) // 3
        t2 = (lo + 2 * hi) // 3
        if t2 - t < 2:
            return []
        return [(lo, t), (t2, hi)]
    if len(sruns) < k:
        return []
    if len(sruns) == k:
        return sruns
    targets = [span * (i + 1) / (k + 1) for i in range(k)]
    centers = [(a + b) * 0.5 for a, b in sruns]
    used = [False] * len(sruns)
    chosen: List[Tuple[int, int]] = []
    for t in targets:
        best_j = -1
        best_d = 1e18
        for j in range(len(sruns)):
            if used[j]:
                continue
            d = abs(centers[j] - t)
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            return []
        used[best_j] = True
        chosen.append(sruns[best_j])
    return sorted(chosen, key=lambda r: r[0])


def gutter_pairs_to_lines(pairs: List[Tuple[int, int]]) -> List[int]:
    if not pairs:
        return []
    return sorted(c for a, b in pairs for c in (a, b))


def detect_montage_gutters(
    im: Image.Image,
    nrows: int,
    ncols: int,
    white_thr: int = 235,
    min_frac: float = 0.94,
    min_run: int = 1,
) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
    """
    Find (ncols-1) full-height and (nrows-1) full-width white *gutter* bands (table layout),
    return the inner edges as vertical and horizontal line lists.
    """
    if np is None:  # pragma: no cover
        return (None, None, "NumPy is required. Run: pip install numpy")
    w, h_ = im.size
    ncg, nrg = max(0, ncols - 1), max(0, nrows - 1)
    if w < 8 or h_ < 8:
        return (None, None, "Image is too small.")
    if nrows < 1 or ncols < 1 or nrows > 10 or ncols > 10:
        return (None, None, "Table size must be 1 to 10 rows and columns.")
    if ncg == 0 and nrg == 0:
        return ([], [], "")

    nmax = 1800
    n_w, n_h = w, h_
    if max(w, h_) > nmax:
        s = nmax / max(w, h_)
        n_w, n_h = max(1, int(w * s)), max(1, int(h_ * s))
        small = im.resize((n_w, n_h), Image.Resampling.BILINEAR).convert("RGB")
    else:
        small = im.convert("RGB")
    rgb = np.array(small, dtype=np.uint8)
    m2 = (rgb[:, :, 0] >= white_thr) & (rgb[:, :, 1] >= white_thr) & (rgb[:, :, 2] >= white_thr)
    h2, w2 = m2.shape[0], m2.shape[1]
    v_gut = (m2.sum(axis=0) / h2) >= min_frac
    h_gut = (m2.sum(axis=1) / w2) >= min_frac
    v_runs = _refine_runs_1d(v_gut, min_run)
    h_runs = _refine_runs_1d(h_gut, min_run)
    mrg, mrh = max(1, int(0.08 * w2)), max(1, int(0.08 * h2))
    v_cand = [r for r in v_runs if mrg <= 0.5 * (r[0] + r[1] - 1) <= w2 - 1 - mrg]
    h_cand = [r for r in h_runs if mrh <= 0.5 * (r[0] + r[1] - 1) <= h2 - 1 - mrh]
    v_pairs: List[Tuple[int, int]] = []
    h_pairs: List[Tuple[int, int]] = []
    if ncg > 0:
        v_sel = _pick_k_gutter_runs(v_cand, w2, ncg)
        if len(v_sel) < ncg:
            return (
                None,
                None,
                f"Need {ncg} good vertical white gap(s) (for a {ncols} column table). "
                "Try a brighter, flat gap or place lines by hand.",
            )
        for p in v_sel:
            v_pairs.append(
                _scale_gutter_pair(p, w2, w) if w2 != w else p
            )
    if nrg > 0:
        h_sel = _pick_k_gutter_runs(h_cand, h2, nrg)
        if len(h_sel) < nrg:
            return (
                None,
                None,
                f"Need {nrg} good horizontal white gap(s) (for a {nrows} row table).",
            )
        for p in h_sel:
            h_pairs.append(
                _scale_gutter_pair(p, h2, h_) if h2 != h_ else p
            )
    v_lines = gutter_pairs_to_lines(v_pairs)
    h_lines = gutter_pairs_to_lines(h_pairs)
    _, _, ok2 = _montage_cell_bands(
        w, h_, nrows, ncols, sorted(v_lines), sorted(h_lines)
    )
    if not ok2:
        return (None, None, "Gutter layout was inconsistent. Try another size or draw lines by hand.")
    return (v_lines, h_lines, "")


def compute_tiles_montage(
    image: Image.Image,
    v_lines: List[int],
    h_lines: List[int],
    nrows: int,
    ncols: int,
) -> List[Tuple[Image.Image, int, int]]:
    w, h_ = image.size
    xb, yb, ok = _montage_cell_bands(
        w, h_, nrows, ncols, sorted(v_lines), sorted(h_lines)
    )
    if not ok:
        return []
    out: List[Tuple[Image.Image, int, int]] = []
    for r, (ys0, ys1) in enumerate(yb):
        for c, (xs0, xs1) in enumerate(xb):
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            out.append((image.crop((xs0, ys0, xs1, ys1)), r, c))
    return out


def export_tiles_montage(
    image: Image.Image,
    vertical: List[int],
    horizontal: List[int],
    out_dir: str,
    basename: str,
    nrows: int,
    ncols: int,
    file_format: str = "png",
    transparent_circle_cutout: bool = False,
    save_options: Optional[SaveExportOptions] = None,
    ignored: Optional[AbstractSet[Tuple[int, int]]] = None,
) -> Tuple[int, List[str]]:
    """Returns (saved_count, errors); a per-tile save failure is collected, not raised."""
    eff = _effective_format_for_saving(file_format, transparent_circle_cutout)
    ext = _export_ext_for_format(eff)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(basename or "image")[0]
    skip: AbstractSet[Tuple[int, int]] = ignored or frozenset()
    n = 0
    errors: List[str] = []
    for tile, r, c in compute_tiles_montage(image, vertical, horizontal, nrows, ncols):
        if (r, c) in skip:
            continue
        out = tile
        if transparent_circle_cutout and CIRCLE_EXTRACT_OK:
            t2 = extract_circle_rgba(tile)
            if t2 is not None:
                out = t2
        p = os.path.join(out_dir, f"{stem}_r{r}_c{c}.{ext}")
        try:
            save_tile_to_path(p, out, eff, save_options)
            n += 1
        except OSError as err:
            errors.append(f"{os.path.basename(p)}: {err}")
    return n, errors


def convert_files(
    paths: List[str],
    file_format: str,
    options: Optional[SaveExportOptions] = None,
) -> Tuple[int, List[str]]:
    """
    Convert each image at `paths` to `file_format`, saving next to the source
    file with the new extension (auto-save; no per-file dialog).
    Returns (converted_count, errors); each error is "<basename>: <message>".
    """
    ext = _export_ext_for_format(file_format)
    count = 0
    errors: List[str] = []
    for src in paths:
        im, err = load_image_file(src)
        if im is None:
            errors.append(f"{os.path.basename(src)}: {err}")
            continue
        dst = f"{os.path.splitext(src)[0]}.{ext}"
        try:
            save_tile_to_path(dst, im, file_format, options)
            count += 1
        except OSError as save_err:
            errors.append(f"{os.path.basename(src)}: {save_err}")
    return count, errors


def _composite_onto_white_for_analysis(src: Image.Image) -> Image.Image:
    """
    RGB composite on white (same as typical JPEG export) so border trim matches
    what the user sees for transparent or palette images.
    """
    if src.mode in ("RGB", "L", "1"):
        return src.convert("RGB")
    if src.mode == "LA":
        l_ = src.getchannel("L")
        a = src.getchannel("A")
        base = Image.merge("RGB", (l_, l_, l_))
        out = Image.new("RGB", src.size, (255, 255, 255))
        out.paste(base, mask=a)
        return out
    if src.mode == "RGBA":
        r, g, b, a_ = src.split()
        base = Image.merge("RGB", (r, g, b))
        out = Image.new("RGB", src.size, (255, 255, 255))
        out.paste(base, mask=a_)
        return out
    if src.mode == "P" and "transparency" in src.info:
        return _composite_onto_white_for_analysis(src.convert("RGBA"))
    return src.convert("RGB")


def _border_trim_bbox(
    rgb: "np.ndarray",
    color_tolerance: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Inclusive bounding box of pixels that are not the estimated border color.
    (left, top, right, bottom) with right/bottom **inclusive** in pixel indices.
    """
    if np is None:  # pragma: no cover
        return None
    h_, w_ = int(rgb.shape[0]), int(rgb.shape[1])
    if w_ < 1 or h_ < 1:
        return None
    edge = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
    )
    if edge.size < 1:
        return None
    bg = np.median(edge.astype(np.float32), axis=0)
    tol = max(0, int(color_tolerance))
    d = np.max(np.abs(rgb.astype(np.float32) - bg), axis=2) > float(tol)
    if not np.any(d):
        return None
    ys, xs = np.where(d)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1, y1)


def _map_trim_box(
    l: int, t: int, r: int, b: int, w0: int, h0: int, wa: int, ha: int
) -> Tuple[int, int, int, int]:
    """
    Map inclusive bbox (l,t,r,b) on analysis (wa, ha) to a crop box on full
    (w0, h0). Returns (left, top, right, bottom) for PIL crop (right, bottom exclusive).
    """
    l0 = int(math.floor(l * w0 / max(1, wa)))
    t0 = int(math.floor(t * h0 / max(1, ha)))
    r0 = int(min(w0, math.ceil((r + 1) * w0 / max(1, wa))))
    b0 = int(min(h0, math.ceil((b + 1) * h0 / max(1, ha))))
    if r0 - l0 < 1 or b0 - t0 < 1:
        return 0, 0, w0, h0
    return l0, t0, r0, b0


def crop_to_content(
    image: Image.Image,
    margin: int = 12,
    color_tolerance: int = 20,
    max_analysis_dim: int = 2500,
) -> Tuple[Image.Image, str]:
    """
    Trim a wide uniform border (e.g. white) by bounding the non-border pixels
    and cropping, with a margin. Uses NumPy. Returns (image, err); err is non-empty
    on failure, otherwise the (possibly) cropped image.
    """
    if np is None:  # pragma: no cover
        return image, "NumPy is required. Run: pip install numpy"
    w0, h0 = image.size
    if w0 < 2 or h0 < 2:
        return image, "Image is too small to crop."

    comp = _composite_onto_white_for_analysis(image)
    aw, ah = w0, h0
    arr = np.ascontiguousarray(np.array(comp, dtype=np.uint8))
    if max(w0, h0) > int(max_analysis_dim) and int(max_analysis_dim) > 32:
        scale = int(max_analysis_dim) / max(w0, h0)
        nw = max(1, int(w0 * scale))
        nh = max(1, int(h0 * scale))
        small = comp.resize((nw, nh), Image.Resampling.BILINEAR)
        aw, ah = nw, nh
        arr = np.ascontiguousarray(np.array(small, dtype=np.uint8))
    if arr.ndim != 3 or int(arr.shape[2]) < 3:
        return image, "Could not analyze the image (unexpected layout)."

    bb = _border_trim_bbox(arr, color_tolerance)
    if bb is None:
        return (
            image,
            "Could not find a clear border (nothing differs enough from the edge color).",
        )

    x0, y0, x1, y1 = bb
    l, t, r, b = _map_trim_box(x0, y0, x1, y1, w0, h0, aw, ah)
    m = max(0, int(margin))
    l2 = max(0, l - m)
    t2 = max(0, t - m)
    r2 = min(w0, r + m)
    b2 = min(h0, b + m)
    if r2 - l2 < 2 or b2 - t2 < 2:
        return image, "Crop would be too small."

    if (l2, t2, r2, b2) == (0, 0, w0, h0):
        return (
            image,
            "Border is already as tight as detected (content fills the image).",
        )

    return image.crop((l2, t2, r2, b2)), ""


def make_color_transparent(
    image: Image.Image, key: Tuple[int, int, int], tolerance: int = 20
) -> Tuple[Image.Image, str]:
    """
    Set alpha to 0 on pixels whose RGB is within tolerance of key (L∞ norm).
    Returns (RGBA image, err). err is "no_match" or "all_gone" or a user message, else "".
    """
    if np is None:  # pragma: no cover
        return image, "no_numpy"
    w0, h0 = image.size
    if w0 < 1 or h0 < 1:
        return image, "no_match"
    kr, kg, kb = (max(0, min(255, int(key[0]))), max(0, min(255, int(key[1]))), max(0, min(255, int(key[2]))))
    t = max(0, int(tolerance))
    rgba = np.ascontiguousarray(
        np.array(image.convert("RGBA"), dtype=np.uint8)
    )
    rgb = rgba[:, :, :3].astype(np.float32)
    kf = np.array((kr, kg, kb), dtype=np.float32).reshape(1, 1, 3)
    d = np.max(np.abs(rgb - kf), axis=2)
    orig_a = rgba[:, :, 3].astype(np.uint8)
    match = d <= float(t)
    if not np.any(match):
        return image, "no_match"
    new_a = np.where(match, 0, orig_a)
    if not np.any(new_a > 0):
        return image, "all_gone"
    out = rgba.copy()
    out[:, :, 3] = new_a
    if np.array_equal(rgba, out):
        return image, "no_change"
    return Image.fromarray(out, "RGBA"), ""


class ImageSplitApp:
    PICK_PX = 8  # max distance in screen pixels to select a line
    # Excel-style "Insert Table" / contact sheet, max 10 in either dimension
    TABLE_DIM_MAX = 10
    # Zoom: scale = _fit * _zoom, where _fit = min(avail/iw, avail/ih) to first fit the window
    ZOOM_MIN = 0.25
    ZOOM_MAX = 8.0
    WHEEL_ZOOM = 1.1
    # Avoid allocating enormous PhotoImage / resize (side length cap)
    MAX_VIEW_PX = 12_000

    def __init__(self) -> None:
        self.root = TkinterDnD.Tk() if DND_OK else tk.Tk()
        self.root.title("Image Split")
        self.root.minsize(800, 600)
        # State
        self.pil_image: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._fit: float = 1.0  # "fit in window" scale (depends on image + window size)
        self._zoom: float = 1.0  # user multiplier; 1.0 = fit, >1 = zoomed in on fit
        self.scale: float = 1.0  # = _fit * _zoom, possibly capped (pixels per source pixel)
        # top-left of displayed image in canvas coordinates (pan)
        self._img_x0: float = 0.0
        self._img_y0: float = 0.0
        self.v_lines: List[int] = []
        self.h_lines: List[int] = []
        self.mode = tk.StringVar(value="v")  # v, h, move, delete, circle, ignore, keycolor
        self.ignored_cells: Set[Tuple[int, int]] = set()
        # (kind, current x or y in image space) — which line is being moved
        self._drag: Optional[Tuple[str, int]] = None
        self._path: Optional[str] = None
        self._recenter: bool = False
        # after zoom, apply pan so (u,v) in image lines up with (cx, cy) in canvas
        self._nudge_pending: Optional[Tuple[float, float, float, float]] = None  # (cx, cy, u, v)
        # v/h = gutter *edges*; export is nrows×ncols content cells (gaps not exported)
        self.montage_table: bool = False
        self.montage_nrows: int = 3
        self.montage_ncols: int = 3
        # chroma key: L∞ color distance; same scale as “crop to content” (0–255)
        self._chroma_key_tolerance: int = 20
        # UI
        self._build_ui()
        self.canvas.bind("<Configure>", self._on_configure)
        self.root.bind("<Escape>", lambda _e: self._end_drag())
        # Dark-ish theme for lines visibility on photos
        self.root.configure(bg="#1e1e1e")
        # Global mouse move for line hover when over canvas
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        if DND_OK:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.status.config(
                text="Open an image to start. (Install tkinterdnd2 for drag-and-drop.)"
            )

    def _build_menu(self) -> None:
        mbar = tk.Menu(self.root, tearoff=0)
        file_menu = tk.Menu(mbar, tearoff=0)
        mbar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open image…", command=self._open, accelerator="Ctrl+O")
        file_menu.add_command(label="Export tiles…", command=self._export, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(
            label="Convert files…", command=self._convert_files_dialog, accelerator="Ctrl+Shift+C"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit, accelerator="Ctrl+Q")
        view_menu = tk.Menu(mbar, tearoff=0)
        mbar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom in", command=lambda: self._bump_zoom(self.WHEEL_ZOOM))
        view_menu.add_command(label="Zoom out", command=lambda: self._bump_zoom(1.0 / self.WHEEL_ZOOM))
        view_menu.add_separator()
        view_menu.add_command(label="Fit to window", command=self._fit_view)
        tools_menu = tk.Menu(mbar, tearoff=0)
        mbar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Auto grid (white gap table)…", command=self._auto_montage_gutters
        )
        tools_menu.add_command(
            label="Crop to content (trim border)…", command=self._crop_to_content
        )
        tools_menu.add_command(
            label="Set color-key tolerance (match)…", command=self._chroma_key_tolerance_dialog
        )
        tools_menu.add_command(label="Clear lines", command=self._clear_lines)
        tools_menu.add_command(label="Clear ignored tiles", command=self._clear_ignored)
        help_menu = tk.Menu(mbar, tearoff=0)
        mbar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Image Split", command=self._about)
        self.root.config(menu=mbar)
        self._bind_menu_shortcuts()

    def _bind_menu_shortcuts(self) -> None:
        def w(fn):
            def inner(_e: tk.Event) -> str:
                fn()
                return "break"

            return inner

        self.root.bind_all("<Control-o>", w(self._open))
        self.root.bind_all("<Control-e>", w(self._export))
        self.root.bind_all("<Control-Shift-C>", w(self._convert_files_dialog))
        self.root.bind_all("<Control-q>", w(self._quit))

    def _about(self) -> None:
        messagebox.showinfo(
            "Image Split",
            "Place vertical and horizontal cut lines, then export tiles. "
            "Crop to content (Tools) removes a big uniform border around a logo. "
            "Key color (eyedropper) makes the clicked color transparent; adjust match tolerance in Tools. "
            "The Auto grid (table size in the sidebar) finds full-span light gaps. "
            "Use the View and Tools menus, or the sidebar: export format (PNG, JPEG, WebP) "
            "and quality or PNG compression, remembered separately per format. "
            "Single circle exports use the PNG compression value. "
            "Use Ignore tile (click cell) to mark tiles you do not want exported; they are hatched in the view. "
            "Drag and drop one image to open it here, or several images to batch-convert them "
            "(File > Convert files… works the same way, without drag-and-drop).",
        )

    def _quit(self) -> None:
        self.root.destroy()

    SIDEBAR_WIDTH = 230

    def _make_section(self, parent: tk.Widget, title: str, start_open: bool = True) -> ttk.Frame:
        """A collapsible sidebar section. Returns the body frame to pack content into."""
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.X, pady=(0, 6))
        body = ttk.Frame(outer, padding=(10, 6, 10, 10))
        state = {"open": start_open}

        def toggle() -> None:
            state["open"] = not state["open"]
            header_btn.config(text=f"{'▾' if state['open'] else '▸'}  {title}")
            if state["open"]:
                body.pack(fill=tk.X)
            else:
                body.pack_forget()

        header_btn = ttk.Label(
            outer,
            text=f"{'▾' if start_open else '▸'}  {title}",
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            padding=(2, 4),
        )
        header_btn.pack(fill=tk.X)
        header_btn.bind("<Button-1>", lambda _e: toggle())
        if start_open:
            body.pack(fill=tk.X)
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 0))
        return body

    def _is_in_sidebar(self, widget: object) -> bool:
        sc = getattr(self, "_sidebar_canvas", None)
        if sc is None or widget is None:
            return False
        return str(widget).startswith(str(sc))

    def _build_ui(self) -> None:
        self._export_format = tk.StringVar(value="WebP")
        self._build_menu()

        self.root.geometry("1180x760")
        self.root.minsize(880, 600)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(main, text="Open an image to start.")
        self.status.pack(anchor=tk.W, side=tk.BOTTOM, pady=(6, 0))

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        sidebar_outer = ttk.Frame(paned)
        paned.add(sidebar_outer, weight=0)
        cframe = ttk.Frame(paned)
        paned.add(cframe, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_outer, width=self.SIDEBAR_WIDTH, highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(
            sidebar_outer, orient=tk.VERTICAL, command=sidebar_canvas.yview
        )
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._sidebar_canvas = sidebar_canvas

        sidebar = ttk.Frame(sidebar_canvas)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _sync_sidebar_scrollregion(_evt: Optional[tk.Event] = None) -> None:
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def _sync_sidebar_width(evt: tk.Event) -> None:
            sidebar_canvas.itemconfigure(sidebar_window, width=evt.width)

        sidebar.bind("<Configure>", _sync_sidebar_scrollregion)
        sidebar_canvas.bind("<Configure>", _sync_sidebar_width)

        # --- File ---
        file_box = self._make_section(sidebar, "File")
        ttk.Button(file_box, text="Open image…", command=self._open).pack(fill=tk.X, pady=2)
        ttk.Button(file_box, text="Export tiles…", command=self._export).pack(fill=tk.X, pady=2)
        ttk.Button(
            file_box, text="Convert files…", command=self._convert_files_dialog
        ).pack(fill=tk.X, pady=2)

        # --- Edit mode (drawing tool) ---
        mode_box = self._make_section(sidebar, "Edit mode")
        for label, value in (
            ("Add vertical line", "v"),
            ("Add horizontal line", "h"),
            ("Move line", "move"),
            ("Delete (click line)", "delete"),
            ("Extract circle (click cell)", "circle"),
            ("Ignore tile (click cell)", "ignore"),
            ("Key color → transparent", "keycolor"),
        ):
            ttk.Radiobutton(mode_box, text=label, variable=self.mode, value=value).pack(
                anchor=tk.W, pady=1
            )
        ttk.Button(mode_box, text="Clear lines", command=self._clear_lines).pack(
            fill=tk.X, pady=(6, 0)
        )
        ttk.Button(mode_box, text="Clear ignored tiles", command=self._clear_ignored).pack(
            fill=tk.X, pady=(2, 0)
        )

        # --- Grid & table ---
        grid_box = self._make_section(sidebar, "Grid & table")
        tr = tuple(str(i) for i in range(1, self.TABLE_DIM_MAX + 1))
        self._table_rows = tk.StringVar(value="3")
        self._table_cols = tk.StringVar(value="3")
        dims_row = ttk.Frame(grid_box)
        dims_row.pack(fill=tk.X)
        ttk.Label(dims_row, text="Rows").pack(side=tk.LEFT)
        self._table_row_combo = ttk.Combobox(
            dims_row, textvariable=self._table_rows, state="readonly", width=3, values=tr
        )
        self._table_row_combo.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(dims_row, text="Cols").pack(side=tk.LEFT)
        self._table_col_combo = ttk.Combobox(
            dims_row, textvariable=self._table_cols, state="readonly", width=3, values=tr
        )
        self._table_col_combo.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(
            grid_box, text="Auto grid (gaps)…", command=self._auto_montage_gutters
        ).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(
            grid_box, text="Crop to content…", command=self._crop_to_content
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            grid_box, text="Color-key tolerance…", command=self._chroma_key_tolerance_dialog
        ).pack(fill=tk.X, pady=2)

        # --- Export settings ---
        export_box = self._make_section(sidebar, "Export settings")
        fmt_row = ttk.Frame(export_box)
        fmt_row.pack(fill=tk.X)
        ttk.Label(fmt_row, text="Format").pack(side=tk.LEFT)
        self._export_combo = ttk.Combobox(
            fmt_row,
            textvariable=self._export_format,
            state="readonly",
            width=8,
            values=("PNG", "JPEG", "WebP"),
        )
        self._export_combo.pack(side=tk.LEFT, padx=(4, 0))

        self._quality_mem: Dict[str, int] = {"png": 6, "jpeg": 92, "webp": 90}
        self._prev_file_fmt: Optional[str] = None
        self._export_quality = tk.IntVar(value=6)
        q_row = ttk.Frame(export_box)
        q_row.pack(fill=tk.X, pady=(6, 0))
        self._q_label = ttk.Label(q_row, text="")
        self._q_label.pack(side=tk.LEFT)
        self._q_spin = ttk.Spinbox(q_row, textvariable=self._export_quality, width=5)
        self._q_spin.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(export_box, text="(remembered per format)", foreground="gray").pack(
            anchor=tk.W, pady=(2, 0)
        )

        self._export_transparent_circles = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            export_box,
            text="Round cutout (transparent)",
            variable=self._export_transparent_circles,
        ).pack(anchor=tk.W, pady=(8, 0))
        self._export_combo.bind("<<ComboboxSelected>>", self._on_export_format_changed)

        # --- View ---
        view_box = self._make_section(sidebar, "View", start_open=False)
        zoom_row = ttk.Frame(view_box)
        zoom_row.pack(fill=tk.X)
        ttk.Button(
            zoom_row, text="−", width=3, command=lambda: self._bump_zoom(1.0 / self.WHEEL_ZOOM)
        ).pack(side=tk.LEFT)
        ttk.Button(
            zoom_row, text="+", width=3, command=lambda: self._bump_zoom(self.WHEEL_ZOOM)
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(zoom_row, text="Fit to window", command=self._fit_view).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Label(
            view_box,
            text="Scroll wheel to zoom, scrollbars to pan when zoomed.",
            foreground="gray",
            wraplength=self.SIDEBAR_WIDTH - 20,
        ).pack(anchor=tk.W, pady=(6, 0))

        self._on_export_format_changed()

        # Canvas in a frame for border
        self.canvas = tk.Canvas(cframe, background="#2d2d2d", highlightthickness=0)
        self._vscroll = ttk.Scrollbar(cframe, orient=tk.VERTICAL, command=self.canvas.yview)
        self._hscroll = ttk.Scrollbar(cframe, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self._hscroll.set, yscrollcommand=self._vscroll.set)
        self._hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self._vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._win_w = 0
        self._win_h = 0
        self._ox = 0.0
        self._oy = 0.0
        self._bind_zoom_wheel()

    def _on_configure(self, _evt: Optional[tk.Event] = None) -> None:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return
        self._win_w, self._win_h = w, h
        if self.pil_image is not None:
            self._rebuild_display()
        else:
            self._draw_idle()

    def _draw_idle(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self._win_w // 2,
            self._win_h // 2,
            text="Open an image (File menu or sidebar), then add or auto-detect cut lines.\n"
            "Drag in Move mode to adjust. Use the mouse wheel or View menu to zoom.\n"
            "Or drop image file(s) here: one opens for editing, several open batch convert.",
            fill="#888",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )

    def _apply_scale_and_pan(self) -> Tuple[float, float, int, int]:
        """Set self._fit, self._zoom, self.scale from self._win and image. Returns (d_wf, d_hf, dw, dh) display sizes."""
        if self.pil_image is None:
            return 0, 0, 0, 0
        iw, ih = self.pil_image.size
        margin = 16
        aw = max(1, self._win_w - 2 * margin)
        ah = max(1, self._win_h - 2 * margin)
        self._fit = min(aw / max(iw, 1), ah / max(ih, 1))
        self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom))
        self.scale = self._fit * self._zoom
        # hard cap: avoid OOM and slow resizes on extreme zoom
        s_cap = min(self.MAX_VIEW_PX / max(iw, 1), self.MAX_VIEW_PX / max(ih, 1), self.scale)
        if s_cap < self.scale:
            self.scale = s_cap
            if self._fit > 0:
                self._zoom = self.scale / self._fit
        dwf = max(1.0, self.scale * iw)
        dhf = max(1.0, self.scale * ih)
        dw, dh = int(dwf), int(dhf)
        if dw < 1:
            dw = 1
        if dh < 1:
            dh = 1
        return dwf, dhf, dw, dh

    def _rebuild_display(self) -> None:
        if self.pil_image is None:
            return
        dwf, dhf, dw, dh = self._apply_scale_and_pan()
        wn, hn = self._win_w, self._win_h
        if self._recenter:
            self._recenter = False
            self._img_x0 = (wn - dwf) / 2.0
            self._img_y0 = (hn - dhf) / 2.0
        elif self._nudge_pending is not None:
            cx, cy, u, v = self._nudge_pending
            self._nudge_pending = None
            self._img_x0 = cx - u * dwf
            self._img_y0 = cy - v * dhf
        self._clamp_pan(dwf, dhf, wn, hn)
        self._ox, self._oy = self._img_x0, self._img_y0
        # Resize for display (integer size)
        small = self.pil_image.resize((dw, dh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(small)
        self.canvas.delete("all")
        self.canvas.create_image(self._ox, self._oy, anchor=tk.NW, image=self._photo)
        self._draw_overlays()
        r_w = max(wn, int(self._ox + dw) + 24, 8)
        r_h = max(hn, int(self._oy + dh) + 24, 8)
        self.canvas.config(scrollregion=(0, 0, r_w, r_h))
        self._update_status()

    def _clamp_pan(self, dwf: float, dhf: float, wn: int, hn: int) -> None:
        if dwf <= wn + 0.1:
            self._img_x0 = (wn - dwf) / 2.0
        else:
            self._img_x0 = min(0, max(self._img_x0, wn - dwf))
        if dhf <= hn + 0.1:
            self._img_y0 = (hn - dhf) / 2.0
        else:
            self._img_y0 = min(0, max(self._img_y0, hn - dhf))

    def _do_zoom_around(self, factor: float, ex: int, ey: int) -> None:
        if self.pil_image is None or self._win_w < 2:
            return
        iw, ih = self.pil_image.size
        dwo = max(1e-6, iw * self.scale)
        dho = max(1e-6, ih * self.scale)
        cx, cy = self.canvas.canvasx(ex), self.canvas.canvasy(ey)
        u = (cx - self._img_x0) / dwo
        v = (cy - self._img_y0) / dho
        in_image = (self._img_x0 <= cx <= self._img_x0 + dwo) and (self._img_y0 <= cy <= self._img_y0 + dho)
        if in_image:
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
        else:
            u, v = 0.5, 0.5
        new_z = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if abs(new_z - self._zoom) < 1e-6:
            return
        self._nudge_pending = (cx, cy, u, v)
        self._zoom = new_z
        self._rebuild_display()

    def _bump_zoom(self, factor: float) -> None:
        w = self.canvas.winfo_width() or 1
        h2 = self.canvas.winfo_height() or 1
        self._do_zoom_around(factor, w // 2, h2 // 2)

    def _fit_view(self) -> None:
        if self.pil_image is None:
            return
        self._zoom = 1.0
        self._nudge_pending = None
        self._recenter = True
        self._rebuild_display()

    @staticmethod
    def _parse_wheel(event: tk.Event) -> Optional[float]:
        n = getattr(event, "num", 0) or 0
        if n in (4, 5):
            return ImageSplitApp.WHEEL_ZOOM if n == 4 else 1.0 / ImageSplitApp.WHEEL_ZOOM
        d = getattr(event, "delta", 0)
        if d == 0:
            return None
        if d > 0:
            return ImageSplitApp.WHEEL_ZOOM
        return 1.0 / ImageSplitApp.WHEEL_ZOOM

    def _on_wheel(self, e: tk.Event) -> str | None:
        t = self.root.winfo_containing(e.x_root, e.y_root)
        if self._is_in_sidebar(t):
            f = self._parse_wheel(e)
            if f is not None:
                self._sidebar_canvas.yview_scroll(-1 if f > 1.0 else 1, "units")
            return "break" if sys.platform == "win32" else None
        if self.pil_image is None:
            return None
        if t is not self.canvas and e.widget is not self.canvas:
            return None
        f = self._parse_wheel(e)
        if f is None:
            return None
        # Coordinates must be relative to the canvas (bind_all on Windows has wrong e.x / e.y)
        ax = e.x_root - self.canvas.winfo_rootx()
        ay = e.y_root - self.canvas.winfo_rooty()
        self._do_zoom_around(f, ax, ay)
        if sys.platform == "win32":
            return "break"
        return None

    def _bind_zoom_wheel(self) -> None:
        if sys.platform == "win32":
            self.root.bind_all("<MouseWheel>", self._on_wheel, add="")
        else:
            self.canvas.bind("<MouseWheel>", self._on_wheel, add="")
        self.canvas.bind("<Button-4>", self._on_wheel, add="")
        self.canvas.bind("<Button-5>", self._on_wheel, add="")
        self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set(), add="")

    def _ix_to_cx(self, ix: int) -> float:
        return self._ox + ix * self.scale

    def _iy_to_cy(self, iy: int) -> float:
        return self._oy + iy * self.scale

    def _cx_to_ix(self, cx: float) -> int:
        return int(round((float(cx) - self._ox) / self.scale))

    def _cy_to_iy(self, cy: float) -> int:
        return int(round((float(cy) - self._oy) / self.scale))

    def _draw_overlays(self) -> None:
        if self.pil_image is None:
            return
        w, h = self.pil_image.size
        color_v = "#00c8ff"
        color_h = "#ff6b4a"
        for ix in self.v_lines:
            x0 = x1 = self._ix_to_cx(ix)
            y0, y1 = self._oy, self._oy + h * self.scale
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill=color_v, tags=("line", f"v{ix}"))

        for iy in self.h_lines:
            y0 = y1 = self._iy_to_cy(iy)
            x0, x1 = self._ox, self._ox + w * self.scale
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill=color_h, tags=("line", f"h{iy}"))

        self._prune_ignored_cells()
        for (x0, y0, x1, y1), r, c in list_tile_rects(
            self.pil_image,
            self.v_lines,
            self.h_lines,
            self.montage_table and self._montage_gutter_shape_ok(),
            self.montage_nrows,
            self.montage_ncols,
        ):
            if (r, c) not in self.ignored_cells:
                continue
            cx0 = self._ix_to_cx(x0)
            cy0 = self._iy_to_cy(y0)
            cx1 = self._ix_to_cx(x1)
            cy1 = self._iy_to_cy(y1)
            if cx0 > cx1:
                cx0, cx1 = cx1, cx0
            if cy0 > cy1:
                cy0, cy1 = cy1, cy0
            self.canvas.create_rectangle(
                cx0,
                cy0,
                cx1,
                cy1,
                fill="#1e1e1e",
                stipple="gray25",
                outline="#8a8a8a",
                width=1,
                tags="ignored",
            )

    def _prune_ignored_cells(self) -> None:
        if not self.ignored_cells or self.pil_image is None:
            return
        rects = list_tile_rects(
            self.pil_image,
            self.v_lines,
            self.h_lines,
            self.montage_table and self._montage_gutter_shape_ok(),
            self.montage_nrows,
            self.montage_ncols,
        )
        valid = {(r, c) for _b, r, c in rects}
        self.ignored_cells &= valid

    def _clear_ignored(self) -> None:
        if not self.ignored_cells:
            return
        self.ignored_cells.clear()
        if self.pil_image is not None:
            self._rebuild_display()
        self.status.config(text="Cleared ignored tile marks.")

    def _pick_line(self, cx: float, cy: float) -> Optional[Tuple[str, int, int]]:
        """
        Return (kind, index, pixel) for the nearest grabbable line, or None.
        Distance is in screen pixels so zoom does not change grab tolerance.
        """
        if self.pil_image is None:
            return None
        w, h = self.pil_image.size
        img_x0, img_x1 = self._ox, self._ox + w * self.scale
        img_y0, img_y1 = self._oy, self._oy + h * self.scale
        p = self.PICK_PX

        best: Optional[Tuple[str, int, int, float]] = None
        for i, xv in enumerate(self.v_lines):
            if not (img_y0 - p <= cy <= img_y1 + p):
                continue
            d = abs(cx - self._ix_to_cx(xv))
            if d > p:
                continue
            if best is None or d < best[3]:
                best = ("v", i, xv, d)
        for i, yh in enumerate(self.h_lines):
            if not (img_x0 - p <= cx <= img_x1 + p):
                continue
            d = abs(cy - self._iy_to_cy(yh))
            if d > p:
                continue
            if best is None or d < best[3]:
                best = ("h", i, yh, d)
        if best is None:
            return None
        return (best[0], best[1], best[2])

    def _on_canvas_motion(self, e: tk.Event) -> None:
        self.canvas.config(cursor="")
        if self.pil_image is None:
            return
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        m = self.mode.get()
        if m == "circle":
            self.canvas.config(cursor="cross")
            return
        if m == "keycolor":
            self.canvas.config(cursor="cross")
            return
        if m == "ignore":
            self.canvas.config(cursor="target")
            return
        p = self._pick_line(cx, cy)
        if p and m in ("move", "delete"):
            self.canvas.config(cursor="fleur" if m == "move" else "X_cursor")
        elif p:
            self.canvas.config(cursor="hand2")

    def _add_unique(self, kind: str, value: int) -> None:
        self.montage_table = False
        w, h = self.pil_image.size  # type: ignore[union-attr]
        if kind == "v":
            v = max(1, min(w - 1, value))
            if any(abs(v - x) < 1.0 for x in self.v_lines):
                return
            self.v_lines.append(v)
            self.v_lines.sort()
        else:
            v = max(1, min(h - 1, value))
            if any(abs(v - y) < 1.0 for y in self.h_lines):
                return
            self.h_lines.append(v)
            self.h_lines.sort()
        self._rebuild_display()

    def _on_press(self, e: tk.Event) -> None:
        if self.pil_image is None:
            return
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        w, h = self.pil_image.size
        ix = self._cx_to_ix(cx)
        iy = self._cy_to_iy(cy)
        if not (0 <= ix < w and 0 <= iy < h):
            return

        mode = self.mode.get()
        if mode == "keycolor":
            if np is None:
                messagebox.showwarning(
                    "Color key", "NumPy is required. Run: pip install numpy"
                )
                return
            rgba0 = self.pil_image.convert("RGBA")
            r_samp, g_samp, b_samp, a_samp = rgba0.getpixel((ix, iy))
            if a_samp < 16:
                messagebox.showinfo(
                    "Color key",
                    "This pixel is nearly or fully transparent. "
                    "Click a solid part of the background to sample that color (not the logo).",
                )
                return
            new_im, err = make_color_transparent(
                self.pil_image, (r_samp, g_samp, b_samp), self._chroma_key_tolerance
            )
            if err == "all_gone":
                messagebox.showwarning(
                    "Color key",
                    "That would make the whole image transparent. Lower the tolerance in Tools, "
                    "or the logo may match the key color too closely (try a different key point on the background).",
                )
                return
            if err == "no_match":
                messagebox.showinfo(
                    "Color key",
                    "No pixels matched. Raise the color-key tolerance (Tools menu) or try another pixel on the background.",
                )
                return
            if err == "no_change":
                messagebox.showinfo(
                    "Color key",
                    "No change — the matching pixels are already fully transparent.",
                )
                return
            if err:
                if err == "no_numpy":
                    messagebox.showwarning(
                        "Color key", "NumPy is required. Run: pip install numpy"
                    )
                return
            self.pil_image = new_im
            self._rebuild_display()
            self.status.config(
                text=(
                    f"Key color → transparent: RGB ({r_samp}, {g_samp}, {b_samp})  |  "
                    f"tolerance {self._chroma_key_tolerance}"
                )
            )
            return

        if mode == "circle":
            if not CIRCLE_EXTRACT_OK:
                messagebox.showwarning(
                    "Extract circle",
                    "OpenCV is not available. Run: pip install opencv-python-headless",
                )
                return
            g = get_tile_at_pixel(
                self.pil_image,
                self.v_lines,
                self.h_lines,
                ix,
                iy,
                self.montage_table and self._montage_gutter_shape_ok(),
                self.montage_nrows,
                self.montage_ncols,
            )
            if g is None:
                messagebox.showinfo(
                    "Extract circle",
                    "Could not map the click to a cell. Add cut lines so each cell has one disc.",
                )
                return
            tile, r, c = g
            out = extract_circle_rgba(tile)
            if out is None:
                messagebox.showwarning(
                    "Extract circle",
                    "No circle was detected in this cell. Check contrast or place lines closer around the disc.",
                )
                return
            path = filedialog.asksaveasfilename(
                title="Save circle (transparent PNG)",
                defaultextension=".png",
                filetypes=[("PNG", "*.png")],
                initialfile=f"circle_r{r}_c{c}.png",
            )
            if path:
                o = self._get_save_options()
                cl = o.png_compress
                out.save(
                    path,
                    format="PNG",
                    compress_level=cl,
                    optimize=True,
                )
            return

        if mode == "ignore":
            g = get_tile_at_pixel(
                self.pil_image,
                self.v_lines,
                self.h_lines,
                ix,
                iy,
                self.montage_table and self._montage_gutter_shape_ok(),
                self.montage_nrows,
                self.montage_ncols,
            )
            if g is None:
                messagebox.showinfo(
                    "Ignore tile",
                    "Click inside a region bounded by your cut lines (or the whole image if there are no lines).",
                )
                return
            _tile, r, c = g
            key = (r, c)
            if key in self.ignored_cells:
                self.ignored_cells.discard(key)
            else:
                self.ignored_cells.add(key)
            self._rebuild_display()
            return

        p = self._pick_line(cx, cy)

        if mode == "delete" and p:
            kind, i, _ = p[0], p[1], p[2]
            if kind == "v" and 0 <= i < len(self.v_lines):
                self.v_lines.pop(i)
            elif kind == "h" and 0 <= i < len(self.h_lines):
                self.h_lines.pop(i)
            self._check_montage_still_valid()
            self._rebuild_display()
            return
        if mode == "move" and p:
            k, _i, val = p
            self._drag = (k, val)
            return
        if mode == "v":
            self._add_unique("v", ix)
        elif mode == "h":
            self._add_unique("h", iy)

    def _on_drag(self, e: tk.Event) -> None:
        if self._drag is None or self.pil_image is None:
            return
        kind, current = self._drag
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        iw, ih = self.pil_image.size
        if kind == "v":
            new_v = self._cx_to_ix(cx)
            new_v = max(1, min(iw - 1, new_v))
            for j, v in enumerate(self.v_lines):
                if v == current:
                    self.v_lines[j] = new_v
                    self._drag = ("v", new_v)
                    break
        else:
            new_h = self._cy_to_iy(cy)
            new_h = max(1, min(ih - 1, new_h))
            for j, yv in enumerate(self.h_lines):
                if yv == current:
                    self.h_lines[j] = new_h
                    self._drag = ("h", new_h)
                    break
        self._rebuild_display()

    def _end_drag(self) -> None:
        if self.pil_image and self._drag is not None:
            self.v_lines.sort()
            self.h_lines.sort()
        self._drag = None
        if self.pil_image:
            self._rebuild_display()

    def _on_release(self, _e: tk.Event) -> None:
        self._end_drag()

    def _table_dims(self) -> Tuple[int, int]:
        try:
            r = int(self._table_rows.get().strip())  # type: ignore[union-attr]
            c = int(self._table_cols.get().strip())  # type: ignore[union-attr]
        except (TypeError, ValueError, AttributeError, tk.TclError):
            r, c = 3, 3
        r = max(1, min(self.TABLE_DIM_MAX, r))
        c = max(1, min(self.TABLE_DIM_MAX, c))
        return r, c

    def _montage_gutter_shape_ok(self) -> bool:
        if not self.montage_table or self.pil_image is None:
            return False
        return (
            len(self.v_lines) == _expected_gutter_line_count(self.montage_ncols)
            and len(self.h_lines) == _expected_gutter_line_count(self.montage_nrows)
        )

    def _check_montage_still_valid(self) -> None:
        if not self.montage_table:
            return
        if not self._montage_gutter_shape_ok():
            self.montage_table = False

    def _auto_montage_gutters(self) -> None:
        if self.pil_image is None:
            messagebox.showinfo("Auto grid", "Open an image first.")
            return
        nrows, ncols = self._table_dims()
        gvx, ghy, err = detect_montage_gutters(self.pil_image, nrows, ncols)
        if gvx is None or ghy is None or err:
            messagebox.showwarning("Auto grid", err or "Detection failed.")
            return
        self.v_lines, self.h_lines = gvx, ghy
        self.montage_nrows, self.montage_ncols = nrows, ncols
        self.montage_table = True
        self._rebuild_display()

    def _crop_to_content(self) -> None:
        if self.pil_image is None:
            messagebox.showinfo("Crop to content", "Open an image first.")
            return
        m = simpledialog.askinteger(
            "Crop to content",
            "Extra margin around the detected content (pixels):",
            minvalue=0,
            maxvalue=500,
            initialvalue=12,
            parent=self.root,
        )
        if m is None:
            return
        old_w, old_h = self.pil_image.size
        new_im, err = crop_to_content(self.pil_image, margin=int(m), color_tolerance=20)
        if err:
            messagebox.showwarning("Crop to content", err)
            return
        new_w, new_h = new_im.size
        self.pil_image = new_im
        self.v_lines = []
        self.h_lines = []
        self.montage_table = False
        self.ignored_cells.clear()
        self._recenter = True
        self._nudge_pending = None
        self._rebuild_display()
        self.status.config(
            text=f"Cropped: {old_w}×{old_h} → {new_w}×{new_h}  (border trimmed, margin {int(m)} px)"
        )

    def _chroma_key_tolerance_dialog(self) -> None:
        t = simpledialog.askinteger(
            "Color-key tolerance",
            "Each color channel: treat pixels within this distance of the key as matching "
            "(0 = exact only; 20 = default, similar to “crop to content”). "
            "Higher values remove a wider fringe of similar colors (e.g. off-white, JPEG).",
            minvalue=0,
            maxvalue=100,
            initialvalue=self._chroma_key_tolerance,
            parent=self.root,
        )
        if t is None:
            return
        self._chroma_key_tolerance = int(t)
        if self.pil_image is not None:
            self.status.config(
                text=f"Color-key match tolerance: {self._chroma_key_tolerance} (per R, G, and B, max of the three).",
            )

    def _clear_lines(self) -> None:
        self.montage_table = False
        self.ignored_cells.clear()
        self.v_lines = []
        self.h_lines = []
        if self.pil_image is not None:
            self._rebuild_display()
        self.status.config(text="Lines cleared.")

    def _update_status(self) -> None:
        if self.pil_image is None:
            return
        nv, nh = len(self.v_lines), len(self.h_lines)
        z_pct = int(round(100.0 * self._zoom))
        if self.montage_table and self._montage_gutter_shape_ok():
            ncells = self.montage_nrows * self.montage_ncols
            ign = f"  |  {len(self.ignored_cells)} ignored" if self.ignored_cells else ""
            self.status.config(
                text=f"Image: {self.pil_image.size[0]}×{self.pil_image.size[1]}  |  {z_pct}% of fit  |  "
                f"Table {self.montage_nrows}×{self.montage_ncols} (white gap export)  |  {ncells} content cells{ign}"
            )
            return
        cells = (nv + 1) * (nh + 1)
        ign = f"  |  {len(self.ignored_cells)} ignored" if self.ignored_cells else ""
        self.status.config(
            text=f"Image: {self.pil_image.size[0]}×{self.pil_image.size[1]}  |  {z_pct}% of fit  |  {nv} vertical, {nh} horizontal  →  {cells} tile(s){ign}"
        )

    def _open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp;*.tif;*.tiff"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        self._load_into_editor(path)

    def _load_into_editor(self, path: str) -> None:
        im, err = load_image_file(path)
        if im is None:
            messagebox.showerror("Open failed", err or "Could not open image.")
            return
        self.pil_image = im
        self._path = path
        self.v_lines = []
        self.h_lines = []
        self._zoom = 1.0
        self._nudge_pending = None
        self._recenter = True
        self.montage_table = False
        self.ignored_cells.clear()
        self._rebuild_display()

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")

    @staticmethod
    def _parse_dnd_paths(data: str) -> List[str]:
        """Parse a tkdnd drop payload into file paths (handles {C:\\a b\\x.png}-style groups)."""
        paths: List[str] = []
        buf = ""
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                buf = ""
            elif ch == "}":
                in_brace = False
                paths.append(buf)
                buf = ""
            elif ch == " " and not in_brace:
                if buf:
                    paths.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            paths.append(buf)
        return paths

    def _on_drop(self, event: tk.Event) -> None:
        paths = self._parse_dnd_paths(str(getattr(event, "data", "")))
        images = [
            p for p in paths if os.path.splitext(p)[1].lower() in self._IMAGE_EXTS and os.path.isfile(p)
        ]
        if not images:
            messagebox.showinfo("Drop", "No image files were found in the dropped item(s).")
            return
        if len(images) == 1:
            self._load_into_editor(images[0])
        else:
            self._convert_files_dialog(preselected=images)

    def _convert_files_dialog(self, preselected: Optional[List[str]] = None) -> None:
        paths: List[str] = preselected if preselected else []
        if not paths:
            picked = filedialog.askopenfilenames(
                title="Convert images…",
                filetypes=[
                    ("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp;*.tif;*.tiff"),
                    ("All", "*.*"),
                ],
            )
            paths = list(picked)
        if not paths:
            return
        choice = self._ask_convert_format()
        if choice is None:
            return
        fmt, options = choice
        n, errors = convert_files(paths, fmt, options)
        ext = _export_ext_for_format(fmt)
        note = ""
        if errors:
            shown = "\n".join(errors[:5])
            more = f"\n(+{len(errors) - 5} more)" if len(errors) > 5 else ""
            note = f"\n\n{len(errors)} file(s) failed:\n{shown}{more}"
        messagebox.showinfo(
            "Convert files",
            f"Converted {n} of {len(paths)} file(s) to .{ext} (saved next to each source file).{note}",
        )

    def _ask_convert_format(self) -> Optional[Tuple[str, SaveExportOptions]]:
        """Small modal: pick an output format + quality. Returns (format, options), or None if cancelled."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Convert to…")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)

        fmt_var = tk.StringVar(value="PNG")
        q_var = tk.IntVar(value=90)
        result: Dict[str, object] = {}

        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Output format:").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        combo = ttk.Combobox(
            frm, textvariable=fmt_var, state="readonly", width=10, values=("PNG", "JPEG", "WebP")
        )
        combo.grid(row=0, column=1, sticky=tk.W, pady=(0, 6))
        q_label = ttk.Label(frm, text="Quality:")
        q_label.grid(row=1, column=0, sticky=tk.W)
        q_spin = ttk.Spinbox(frm, textvariable=q_var, width=6, from_=0, to=100)
        q_spin.grid(row=1, column=1, sticky=tk.W)

        def sync_quality(_e: Optional[tk.Event] = None) -> None:
            f = fmt_var.get()
            if f == "PNG":
                q_label.config(text="Compression (0–9):")
                q_spin.config(from_=0, to=9)
                if q_var.get() > 9:
                    q_var.set(6)
            elif f == "JPEG":
                q_label.config(text="Quality (1–100):")
                q_spin.config(from_=1, to=100)
                if q_var.get() < 1:
                    q_var.set(92)
            else:
                q_label.config(text="Quality (0–100):")
                q_spin.config(from_=0, to=100)

        combo.bind("<<ComboboxSelected>>", sync_quality)
        sync_quality()

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(12, 0), sticky=tk.E)

        def on_ok() -> None:
            result["fmt"] = fmt_var.get()
            result["q"] = int(q_var.get())
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Convert", command=on_ok).pack(side=tk.RIGHT)

        dlg.wait_window()
        if "fmt" not in result:
            return None
        fmt_key = {"PNG": "png", "JPEG": "jpeg", "WebP": "webp"}[str(result["fmt"])]
        q = int(result["q"])  # type: ignore[arg-type]
        if fmt_key == "png":
            opts = SaveExportOptions(png_compress=self._clamp_int(q, 0, 9))
        elif fmt_key == "jpeg":
            opts = SaveExportOptions(jpeg_quality=self._clamp_int(q, 1, 100))
        else:
            opts = SaveExportOptions(webp_quality=self._clamp_int(q, 0, 100))
        return fmt_key, opts

    def _export_file_format(self) -> str:
        m = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}
        k = (self._export_format.get() or "PNG").strip().upper()
        if k in ("JPG", "JPEG", "JPE"):
            k = "JPEG"
        if k in ("WEBP", "WEPB"):
            k = "WEBP"
        return m.get(k, "png")

    @staticmethod
    def _clamp_int(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    def _flush_current_quality_to_mem(self) -> None:
        k = self._export_file_format()
        try:
            v = int(self._export_quality.get())
        except (TypeError, ValueError, tk.TclError):
            v = self._quality_mem.get(k, 6)
        if k == "png":
            self._quality_mem[k] = self._clamp_int(v, 0, 9)
        elif k == "jpeg":
            self._quality_mem[k] = self._clamp_int(v, 1, 100)
        else:
            self._quality_mem[k] = self._clamp_int(v, 0, 100)
        self._export_quality.set(self._quality_mem[k])

    def _on_export_format_changed(self, _event: Optional[tk.Event] = None) -> None:
        if not hasattr(self, "_q_spin") or self._q_spin is None:
            return
        new = self._export_file_format()
        if self._prev_file_fmt is not None and self._prev_file_fmt != new:
            k_old = self._prev_file_fmt
            try:
                v = int(self._export_quality.get())
            except (TypeError, ValueError, tk.TclError):
                v = self._quality_mem.get(k_old, 6)
            if k_old == "png":
                self._quality_mem[k_old] = self._clamp_int(v, 0, 9)
            elif k_old == "jpeg":
                self._quality_mem[k_old] = self._clamp_int(v, 1, 100)
            else:
                self._quality_mem[k_old] = self._clamp_int(v, 0, 100)
        v_new = self._quality_mem.get(new, 6)
        if new == "png":
            v_new = self._clamp_int(int(v_new), 0, 9)
        elif new == "jpeg":
            v_new = self._clamp_int(int(v_new), 1, 100)
        else:
            v_new = self._clamp_int(int(v_new), 0, 100)
        self._export_quality.set(v_new)
        self._prev_file_fmt = new
        if new == "png":
            self._q_label.config(text="PNG compress (0–9):")
            self._q_spin.config(from_=0, to=9, increment=1)
        elif new == "jpeg":
            self._q_label.config(text="JPEG quality (1–100):")
            self._q_spin.config(from_=1, to=100, increment=1)
        else:
            self._q_label.config(text="WebP quality (0–100):")
            self._q_spin.config(from_=0, to=100, increment=1)

    def _get_save_options(self) -> SaveExportOptions:
        self._flush_current_quality_to_mem()
        j = int(self._quality_mem.get("jpeg", 92))
        wq = int(self._quality_mem.get("webp", 90))
        pc = int(self._quality_mem.get("png", 6))
        return SaveExportOptions(
            jpeg_quality=self._clamp_int(j, 1, 100),
            webp_quality=self._clamp_int(wq, 0, 100),
            png_compress=self._clamp_int(pc, 0, 9),
            webp_method=6,
        )

    def _export(self) -> None:
        if self.pil_image is None:
            messagebox.showinfo("Export", "Open an image first.")
            return
        d = filedialog.askdirectory(title="Export folder")
        if not d:
            return
        self._prune_ignored_cells()
        n_skip = len(self.ignored_cells)
        name = os.path.basename(self._path or "image.png")
        fmt = self._export_file_format()
        tc = self._export_transparent_circles.get()
        eff = _effective_format_for_saving(fmt, tc)
        ext = _export_ext_for_format(eff)
        save_opts = self._get_save_options()
        ign = frozenset(self.ignored_cells)
        try:
            if self.montage_table and self._montage_gutter_shape_ok():
                n, errors = export_tiles_montage(
                    self.pil_image,
                    self.v_lines,
                    self.h_lines,
                    d,
                    name,
                    self.montage_nrows,
                    self.montage_ncols,
                    file_format=fmt,
                    transparent_circle_cutout=tc,
                    save_options=save_opts,
                    ignored=ign,
                )
            else:
                n, errors = export_tiles(
                    self.pil_image,
                    self.v_lines,
                    self.h_lines,
                    d,
                    name,
                    file_format=fmt,
                    transparent_circle_cutout=tc,
                    save_options=save_opts,
                    ignored=ign,
                )
        except OSError as err:
            messagebox.showerror("Export failed", f"Could not create the export folder:\n{err}")
            return
        note = ""
        if tc and (fmt or "").lower() in ("jpg", "jpeg", "jpe"):
            note = "\n\n(Transparent cutouts are saved as PNG; JPEG has no alpha channel.)"
        if n_skip:
            note += f"\n\n({n_skip} tile(s) marked ignored were not exported; click them again in Ignore mode to unmark.)"
        if errors:
            shown = "\n".join(errors[:5])
            more = f"\n(+{len(errors) - 5} more)" if len(errors) > 5 else ""
            note += f"\n\n{len(errors)} tile(s) failed to save:\n{shown}{more}"
        messagebox.showinfo("Export", f"Wrote {n} .{ext} file(s) to:\n{d}{note}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ImageSplitApp().run()
