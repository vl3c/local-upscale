"""Crop a photographed print: find it on the surface it lies on, straighten it, trim its edges.

For phone shots or scans of paper photos, where a table, scanner lid or album page shows around the
print. Run it before `variants`, so the restoration only ever sees the photo itself. Always look at
the crop_check image it writes: on a picture that is not a photographed print it can mistake a
long straight line near the border (a door frame, a horizon) for the print's edge.
"""
from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def _line_through(seg):
    x1, y1, x2, y2 = (float(v) for v in seg)
    return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])  # ax + by + c = 0


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def find_print(img, band: float = 0.10, min_span: float = 0.5):
    """Corners (TL, TR, BR, BL) of the print in full-resolution pixels, or None.

    A photographed print has a long, straight edge along each side where it meets the surface
    under it. For each side, the longest near-straight edge (Canny + probabilistic Hough) inside
    the outer `band` of the picture, whose collinear pieces add up to at least half of `min_span`
    of that side, is taken as the
    print's edge; a side with no such edge is assumed to run along the picture's own border (the
    print fills the shot there). None if no side has an edge."""
    H, W = img.shape[:2]
    k = 1000 / max(H, W)
    small = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=int(min(h, w) * 0.15), maxLineGap=12)
    segs = [] if segs is None else [tuple(int(v) for v in seg.reshape(-1)[:4]) for seg in segs]

    def best(side):
        """The side's edge as a line (a, b, c), from the collinear group of segments with the most
        total length; edges are often broken into pieces by texture or a curled corner."""
        vert = side in ("left", "right")
        span = h if vert else w
        cands = []
        for x1, y1, x2, y2 in segs:
            along, across = (abs(y2 - y1), abs(x2 - x1)) if vert else (abs(x2 - x1), abs(y2 - y1))
            if across > along * 0.18:
                continue
            # where the segment's line crosses the side's midpoint
            if vert:
                t = (span / 2 - y1) / (y2 - y1) if y2 != y1 else 0.0
                pos = x1 + t * (x2 - x1)
                ok = pos < w * band if side == "left" else pos > w * (1 - band)
            else:
                t = (span / 2 - x1) / (x2 - x1) if x2 != x1 else 0.0
                pos = y1 + t * (y2 - y1)
                ok = pos < h * band if side == "top" else pos > h * (1 - band)
            if ok:
                cands.append((pos, along, (x1, y1, x2, y2)))
        if not cands:
            return None
        cands.sort()
        tol = 0.006 * (w if vert else h)
        groups, cur = [], [cands[0]]
        for c in cands[1:]:
            if c[0] - cur[-1][0] <= tol:
                cur.append(c)
            else:
                groups.append(cur)
                cur = [c]
        groups.append(cur)
        long_enough = [grp for grp in groups if sum(c[1] for c in grp) >= min_span * span * 0.5]
        if not long_enough:
            return None
        # The print's edge is the outermost long line; lines further in belong to the picture.
        g = long_enough[0] if side in ("left", "top") else long_enough[-1]
        pts = np.float32([[c[2][0], c[2][1]] for c in g] + [[c[2][2], c[2][3]] for c in g])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([vy, -vx, vx * y0 - vy * x0])

    found = {s: best(s) for s in ("left", "right", "top", "bottom")}
    if all(v is None for v in found.values()):
        return None
    border = {"left": (0, 0, 0, h), "right": (w - 1, 0, w - 1, h), "top": (0, 0, w, 0), "bottom": (0, h - 1, w, h - 1)}
    L = {s: found[s] if found[s] is not None else _line_through(border[s]) for s in found}
    quad = np.float32([_intersect(L["top"], L["left"]), _intersect(L["top"], L["right"]),
                       _intersect(L["bottom"], L["right"]), _intersect(L["bottom"], L["left"])])
    quad[:, 0] = quad[:, 0].clip(0, w - 1)
    quad[:, 1] = quad[:, 1].clip(0, h - 1)
    return quad / k


def straighten(img, corners, inset: float = 0.02) -> np.ndarray:
    """Warp the print to a rectangle and trim `inset` (a fraction of each side) to drop the white
    paper border and any sliver of background left by a curled edge."""
    tl, tr, br, bl = order_corners(corners)
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    flat = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4)
    dx, dy = int(round(w * inset)), int(round(h * inset))
    return flat[dy:h - dy, dx:w - dx]


def preview(img, corners) -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [np.int32(order_corners(corners))], True, (0, 0, 255), max(3, img.shape[1] // 300))
    return out
