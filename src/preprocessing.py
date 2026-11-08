"""Phase A: loading, cleaning and binarising the manuscript image."""
import cv2
import numpy as np

import config


def load_image(path):
    """Read an image as BGR. Raises ValueError for corrupted/unreadable files."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_background(gray):
    """Divide by a heavily blurred copy to flatten uneven lighting and stains."""
    k = (max(gray.shape) // 8) | 1
    bg = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32) + 1.0
    flat = np.clip(gray.astype(np.float32) / bg * 200.0, 0, 255).astype(np.uint8)
    return flat


def enhance(gray):
    """Denoise (edge preserving) and boost local contrast for faded ink."""
    den = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=5, searchWindowSize=15)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=(8, 8))
    return clahe.apply(den)


def binarize(gray):
    """Ink = 255. Adaptive threshold so faint ink survives uneven lighting.

    Hysteresis: a weak (faint) pixel is kept only if its connected stroke also
    contains a strong dark pixel. This keeps faded ink but drops paper/leaf texture."""
    block = max((min(gray.shape) // 12) | 1, 15)
    weak = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, config.THRESH_C)
    local_bg = cv2.GaussianBlur(gray, (block, block), 0).astype(np.float32)
    strong = (gray.astype(np.float32) < config.STRONG_INK_RATIO * local_bg).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(labels[strong > 0])] = True
    has_strong[0] = False
    return (has_strong[labels].astype(np.uint8)) * 255


def remove_ruling_lines(binary):
    """Remove long thin horizontal/vertical lines (page borders, scribe lines)."""
    h, w = binary.shape
    out = binary.copy()
    for size, horizontal in ((max(w // 6, 25), True), (max(h // 5, 25), False)):
        shape = (size, 1) if horizontal else (1, size)
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, shape))
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        out[lines > 0] = 0
    return out


def clean_noise(binary):
    """Small opening removes salt-and-pepper specks without touching strokes."""
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def find_page_mask(gray):
    """Mask of the manuscript leaf/page if it sits on a clearly different background
    (e.g. palm leaf on a coloured cloth). Otherwise the whole image counts as page."""
    h, w = gray.shape
    full = np.full((h, w), 255, np.uint8)
    k = (min(h, w) // 10) | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return full
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    margin_x, margin_y = 0.01 * w, 0.01 * h
    touching = sum([x < margin_x, y < margin_y, x + bw > w - margin_x, y + bh > h - margin_y])
    frac = cv2.contourArea(c) / (h * w)
    if touching >= 2 or not (0.15 < frac < 0.95):
        return full
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(c)], -1, 255, -1)
    return mask


def suppress_background(binary, gray, page_mask):
    """If the page sits on a textured background (cloth, table), keep ink on the page
    but outside it accept only near-black pixels (e.g. printed watermark text)."""
    if page_mask.min() > 0:
        return binary
    k = max(3, min(binary.shape) // 60) | 1
    inside = cv2.erode(page_mask, np.ones((k, k), np.uint8), iterations=2)
    outside_ink = ((gray < config.DARK_PIXEL_MAX) & (page_mask == 0)).astype(np.uint8) * 255
    outside_ink = cv2.morphologyEx(outside_ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.bitwise_or(cv2.bitwise_and(binary, inside), outside_ink)


def estimate_skew(binary):
    """Skew angle (degrees) that maximises horizontal projection sharpness."""
    small = cv2.resize(binary, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-config.MAX_SKEW_DEG, config.MAX_SKEW_DEG + 0.01, 0.5):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST)
        profile = rot.sum(axis=1).astype(np.float64)
        score = np.var(np.diff(profile))
        if score > best_score:
            best_angle, best_score = angle, score
    return float(best_angle)


def preprocess(img):
    """Run all preprocessing steps. Returns a dict of intermediate results."""
    gray = to_gray(img)
    flat = normalize_background(gray)
    enhanced = enhance(flat)
    page_mask = find_page_mask(gray)
    binary = clean_noise(remove_ruling_lines(binarize(enhanced)))
    binary = suppress_background(binary, gray, page_mask)
    return {"gray": gray, "flat": flat, "enhanced": enhanced,
            "binary": binary, "page_mask": page_mask}


def rotation_matrix(shape, angle):
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
