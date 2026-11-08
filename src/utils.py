"""File handling and bounding-box helpers."""
import json
from pathlib import Path

import cv2
import numpy as np

import config


def list_images(folder):
    """Sorted list of supported image files directly inside a folder."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS)


def clip_box(box, width, height):
    """Return [x_min, y_min, x_max, y_max] as ints inside the image, or None if empty."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
    y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def is_valid_region(region, width, height):
    x0, y0, x1, y1 = region["bbox"]
    return (all(isinstance(v, int) for v in region["bbox"])
            and 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            and region["label"] in config.CLASSES
            and 0.0 <= region["confidence"] <= 1.0)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"Could not encode image {path}")
    buf.tofile(str(path))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
