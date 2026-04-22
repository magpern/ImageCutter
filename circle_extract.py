"""
Detect a dominant round graphic in a cell (light background) and return an
RGBA crop with a smooth transparent region outside the circle. Requires
opencv-python-headless and numpy.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

CIRCLE_EXTRACT_OK: bool = cv2 is not None

__all__ = ("CIRCLE_EXTRACT_OK", "extract_circle_rgba")


def _hough_default_circle(gray: np.ndarray) -> Optional[Tuple[float, float, int]]:
    if cv2 is None:  # pragma: no cover
        return None
    h, w = gray.shape[:2]
    side = int(min(h, w))
    blur = cv2.medianBlur(gray, 5) if min(h, w) > 32 else gray
    edges = cv2.Canny(blur, 40, 120)
    minr = max(3, int(side * 0.1))
    maxr = int(side * 0.5)
    min_dist = int(max(8, min(h, w) // 4))
    circles: Any = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        1.2,
        min_dist,
        param1=50,
        param2=25,
        minRadius=minr,
        maxRadius=maxr,
    )
    if circles is None or not len(circles[0]):  # type: ignore[union-attr, arg-type]
        return None
    c = np.round(circles[0][0]).astype(int)
    return (float(c[0]), float(c[1]), int(c[2]))  # x, y, r


def extract_circle_rgba(pil: Image.Image) -> Optional[Image.Image]:
    """
    Fit a circle to the main non-bright area; return a tight-cropped square RGBA
    with a soft 2px alpha falloff at the circle edge. None if this fails.
    """
    if cv2 is None:
        return None
    w, h = pil.size
    if w < 8 or h < 8:
        return None

    rgb = np.ascontiguousarray(pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hh, ww = gray.shape[:2]

    _, th_inv = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY_INV)
    if int(th_inv.max()) < 1:
        _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = 255 - th_inv
    if int(th_inv.max()) < 1:
        hough = _hough_default_circle(gray)
        if hough is None:
            return None
        return _build_rgba(rgb, *hough, ww, hh)

    k = max(3, min(21, 2 * int(0.015 * min(ww, hh)) + 1))
    th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        hough = _hough_default_circle(gray)
        if hough is None:
            return None
        return _build_rgba(rgb, *hough, ww, hh)
    c = max(cnts, key=cv2.contourArea)
    if float(cv2.contourArea(c)) < 20:
        hough = _hough_default_circle(gray)
        if hough is None:
            return None
        return _build_rgba(rgb, *hough, ww, hh)

    (rc_x, rc_y), raw_r = cv2.minEnclosingCircle(c)
    r = int(round(float(raw_r)))
    return _build_rgba(rgb, float(rc_x), float(rc_y), r, ww, hh)


def _build_rgba(
    rgb: np.ndarray, cx: float, cy: float, r: int, ww: int, hh: int
) -> Optional[Image.Image]:
    r = max(2, int(r))
    dlim = (min(cx, cy, ww - 0.5 - cx, hh - 0.5 - cy) - 1.0)
    if dlim < 0:
        return None
    r = min(r, int(dlim))
    if r < 2:
        return None

    yy, xx = np.ogrid[:hh, :ww]
    d = np.hypot(xx - cx, yy - cy)
    t = 1.0 - np.clip((d - (float(r) - 0.5)) / 2.0, 0, 1)
    a = (t * 255.0 + 0.5).astype(np.uint8)
    if int(a.max()) < 1:
        return None

    rgba = np.dstack((rgb, a))
    yx = np.argwhere(a > 8)
    if yx.size == 0:
        return None
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0) + 1
    y0, x0 = max(0, int(y0) - 1), max(0, int(x0) - 1)
    y1, x1 = min(hh, int(y1) + 1), min(ww, int(x1) + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    return Image.fromarray(rgba[y0:y1, x0:x1], "RGBA")
