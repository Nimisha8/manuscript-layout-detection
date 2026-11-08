"""Phase B: turn the binary ink image into candidate text blocks."""
import cv2
import numpy as np

import config
from src import preprocessing as pp
from src.utils import clip_box


def _components(binary, page_area):
    """Connected components, dropping specks and giant objects (not characters)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    min_area = max(8, int(config.MIN_COMPONENT_AREA_FRAC * h * w))
    keep = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area or area > 0.08 * page_area or ch > 0.5 * h:
            continue
        thin = min(cw, ch) <= 4 and max(cw, ch) >= 25   # leftover ruling-line fragments
        if thin and max(cw, ch) / max(1, min(cw, ch)) > 8:
            continue
        keep.append(i)
    return labels, stats, keep


def _unit_size(stats, keep):
    """Typical character height: median height of the larger components."""
    if not keep:
        return 10.0
    areas = np.array([stats[i][4] for i in keep])
    heights = np.array([stats[i][3] for i in keep])
    big = heights[areas >= np.percentile(areas, 40)]
    return float(max(np.median(big), 6.0))


def group_lines(mask, unit, stats=None, keep=None, page_mask=None):
    """Morphological grouping: smear characters horizontally into text lines."""
    kx = max(3, int(config.LINE_DILATE_X * unit))
    ky = max(1, int(config.LINE_DILATE_Y * unit))
    smeared = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    cnts, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ch = h
        if stats is not None:  # typical character height inside this line
            hs = [stats[i][3] for i in keep
                  if x <= stats[i][0] + stats[i][2] / 2 < x + w and y <= stats[i][1] + stats[i][3] / 2 < y + h]
            if hs:
                ch = float(np.median(hs))
        on_page = True if page_mask is None else bool(page_mask[y:y + h, x:x + w].mean() > 127)
        lines.append([x, y, w, h, ch, 1, on_page])  # x, y, w, h, char_height, n_lines, on_page
    return lines


def _hoverlap(a, b):
    inter = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    return max(0, inter) / max(1, min(a[2], b[2]))


def merge_lines(lines, unit):
    """Stack lines that are vertically close and horizontally overlapping into blocks."""
    boxes = [list(b) for b in lines]
    changed = True
    while changed:
        changed = False
        boxes.sort(key=lambda b: b[1])
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                gap = b[1] - (a[1] + a[3])
                max_gap = config.BLOCK_MAX_VGAP * max(2.0 * min(a[4], b[4]), 0.5 * min(a[3], b[3]))
                ratio = a[4] / max(b[4], 1e-6)
                similar_size = 1 / config.MAX_CHAR_SIZE_RATIO <= ratio <= config.MAX_CHAR_SIZE_RATIO
                if a[6] == b[6] and similar_size and gap <= max_gap and _hoverlap(a, b) >= config.BLOCK_MIN_HOVERLAP:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    n = a[5] + b[5]
                    ch = (a[4] * a[5] + b[4] * b[5]) / n
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0, ch, n, a[6]]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def split_columns(block, mask, unit):
    """Vertical projection profile: split a block where an empty vertical band
    runs through its whole height (multi-column layout)."""
    x, y, w, h = block[:4]
    crop = mask[y:y + h, x:x + w]
    profile = (crop > 0).sum(axis=0)
    empty = profile <= max(1, 0.01 * h)
    min_gap = int(config.COLUMN_GAP_UNITS * unit)
    cuts, start = [], None
    for i, e in enumerate(np.append(empty, False)):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_gap and start > 0 and i < w:
                cuts.append((start, i))
            start = None
    if not cuts:
        return [block]
    parts, prev = [], 0
    for a, b in cuts:
        parts.append((prev, a))
        prev = b
    parts.append((prev, w))
    out = []
    for a, b in parts:
        sub = crop[:, a:b]
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            continue
        out.append([x + a + int(xs.min()), y + int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] + list(block[4:]))
    return out or [block]


def block_features(block, mask, flat, page_mask, labels, stats, keep_set):
    """Measurements used later by the classifier."""
    x, y, w, h = block[:4]
    sub_labels = labels[y:y + h, x:x + w]
    ids = [i for i in np.unique(sub_labels) if i in keep_set]
    if not ids:
        return None
    heights = np.array([stats[i][3] for i in ids])
    areas = np.array([stats[i][4] for i in ids])
    ink = (np.isin(sub_labels, ids))
    ink_area = int(ink.sum())
    pad = 0
    # tighten the box: ignore the outermost 1% of ink pixels (stray specks / line fragments)
    ys, xs = np.nonzero(ink)
    if len(xs) > 200:
        lo, hi = config.TRIM_PERCENTILE, 100 - config.TRIM_PERCENTILE
        tx0, tx1 = int(np.percentile(xs, lo)), int(np.percentile(xs, hi)) + 1
        ty0, ty1 = int(np.percentile(ys, lo)), int(np.percentile(ys, hi)) + 1
        pad = int(config.BOX_PAD_CHAR_FRAC * np.median(heights))
        x, y, w, h = x + tx0, y + ty0, tx1 - tx0, ty1 - ty0
        ink = ink[ty0:ty1, tx0:tx1]
        sub_labels = sub_labels[ty0:ty1, tx0:tx1]
        ink_area = int(ink.sum())
    flat_crop = flat[y:y + h, x:x + w].astype(np.float32)
    bg = np.median(flat_crop[~ink]) if (~ink).any() else 200.0
    contrast = float(bg - np.percentile(flat_crop[ink], 25))
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    stroke = float(2 * np.percentile(dt[ink], 90)) if ink.any() else 1.0
    page_crop = page_mask[y:y + h, x:x + w]
    in_page = float((page_crop > 0).mean())
    return {
        "bbox": [x - pad, y - pad, x + w + pad, y + h + pad],
        "ink_area": ink_area,
        "density": ink_area / float(w * h),
        "n_components": len(ids),
        "char_height": float(np.median(heights)),
        "contrast": contrast,
        "stroke_px": stroke,
        "in_page": in_page,
    }


def detect_blocks(pre):
    """Full detection step. Returns (list of feature dicts, skew angle) in original coordinates."""
    binary, flat, page_mask = pre["binary"], pre["flat"], pre["page_mask"]
    h, w = binary.shape

    angle = pp.estimate_skew(binary)
    M = None
    if abs(angle) >= config.MIN_SKEW_TO_CORRECT:
        M = pp.rotation_matrix(binary.shape, angle)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
        flat = cv2.warpAffine(flat, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=200)
        page_mask = cv2.warpAffine(page_mask, M, (w, h), flags=cv2.INTER_NEAREST)

    labels, stats, keep = _components(binary, h * w)
    if not keep:
        return [], angle
    unit = _unit_size(stats, keep)
    keep_set = set(keep)
    mask = np.isin(labels, keep).astype(np.uint8) * 255

    lines = group_lines(mask, unit, stats, keep, page_mask)
    blocks = merge_lines(lines, unit)
    split = []
    for b in blocks:
        split.extend(split_columns(b, mask, unit))

    feats = []
    for b in split:
        f = block_features(b, mask, flat, page_mask, labels, stats, keep_set)
        if f and f["ink_area"] >= config.MIN_BLOCK_INK_FRAC * h * w and f["n_components"] >= 2:
            feats.append(f)

    if M is not None:  # map boxes back to the original (unrotated) image
        inv = cv2.invertAffineTransform(M)
        for f in feats:
            x0, y0, x1, y1 = f["bbox"]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            pts = cv2.transform(pts[None], inv)[0]
            f["bbox"] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
    for f in feats:
        f["bbox"] = clip_box(f["bbox"], w, h)
    feats = [f for f in feats if f["bbox"]]
    f_unit = unit
    for f in feats:
        f["unit"] = f_unit
    return feats, angle
