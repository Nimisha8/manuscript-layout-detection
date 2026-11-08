"""Phase C: rule-based classification of candidate blocks.

Each block is judged relative to the *main text body* of the same page (the block with
the most ink), not by absolute page coordinates, so differently shaped pages work.
"""
import numpy as np

import config


def _clip01(v):
    return float(min(0.98, max(0.05, v)))


def _vertical_overlap(a, b):
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))


def _horizontal_overlap(a, b):
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return max(0, inter) / max(1, min(a[2] - a[0], b[2] - b[0]))


def _pick_main(blocks):
    """Reference body: block with most ink among those sitting on the page itself."""
    on_page = [b for b in blocks if b["in_page"] >= 0.5] or blocks
    return max(on_page, key=lambda b: b["ink_area"])


def classify_block(b, main, img_w, img_h):
    """Return (label, confidence, reason) for one block."""
    x0, y0, x1, y1 = b["bbox"]
    mx0, my0, mx1, my1 = main["bbox"]
    w, h = x1 - x0, y1 - y0
    mw, mh = mx1 - mx0, my1 - my0
    if b is main:
        return "main_text", 0.90, "largest text body"

    small = b["char_height"] / main["char_height"]
    faint = b["contrast"] / max(1.0, main["contrast"])
    thin_strokes = b["stroke_px"] < config.THIN_STROKE_RATIO * main["stroke_px"]
    rel_ink = b["ink_area"] / main["ink_area"]
    hov = _horizontal_overlap(b["bbox"], main["bbox"])
    vov = _vertical_overlap(b["bbox"], main["bbox"])

    # --- side text (checked first: marginalia is often smaller/fainter than the body): next to the body, overlapping its vertical extent ---
    left_of = x1 <= mx0 + 0.05 * mw
    right_of = x0 >= mx1 - 0.05 * mw
    if (left_of or right_of) and vov > 0.3 and w < config.SIDE_MAX_WIDTH_RATIO * mw:
        if b["in_page"] >= 0.5 and faint >= config.FAINT_CONTRAST_RATIO:
            return "side_text", _clip01(0.55 + 0.4 * vov), "narrow block beside the body"

    # --- filler: printed/English overlays, decoration, pencil, text off the page ---
    if b["in_page"] < 0.5:
        return "filler", _clip01(0.85 - 0.2 * b["in_page"]), "outside the page area"
    if small < config.SMALL_TEXT_RATIO and thin_strokes:
        return "filler", _clip01(0.6 + 0.4 * (1 - small)), "small thin-stroke print (English/overlay)"
    if faint < config.FAINT_CONTRAST_RATIO:
        return "filler", _clip01(0.55 + 0.4 * (1 - faint)), "faint ink (pencil/stain)"
    if b["density"] < 0.02 and rel_ink < 0.05:
        return "filler", 0.45, "sparse small mark"

    # --- header / footer: above / below the body, horizontally aligned with it ---
    if y1 <= my0 + 0.1 * unit_of(main) and hov > 0.2:
        small_block = h < 0.6 * mh
        conf = 0.55 + 0.25 * hov + (0.15 if small_block else 0)
        return "header", _clip01(conf), "text block above the body"
    if y0 >= my1 - 0.1 * unit_of(main) and hov > 0.2:
        small_block = h < 0.6 * mh
        conf = 0.55 + 0.25 * hov + (0.15 if small_block else 0)
        return "footer", _clip01(conf), "text block below the body"

    # --- anything substantial that overlaps the body area: another column/paragraph ---
    if rel_ink > 0.1 and (hov > 0.1 or vov > 0.3):
        return "main_text", _clip01(0.5 + 0.4 * min(1.0, rel_ink)), "additional body block"
    return "filler", 0.4, "unclassified small region"


def unit_of(block):
    return block.get("unit", 10.0)


def classify_blocks(blocks, img_w, img_h):
    """Classify every candidate. Returns a list of region dicts."""
    if not blocks:
        return []
    main = _pick_main(blocks)
    # ignore specks (e.g. leftover scribe-line fragments) unless they are off the page
    blocks = [b for b in blocks if b is main or b["in_page"] < 0.5
              or b["ink_area"] >= config.MIN_REL_INK * main["ink_area"]]
    regions = []
    for b in blocks:
        label, conf, reason = classify_block(b, main, img_w, img_h)
        regions.append({"bbox": [int(v) for v in b["bbox"]], "label": label,
                        "confidence": round(conf, 3), "reason": reason})
    return regions
