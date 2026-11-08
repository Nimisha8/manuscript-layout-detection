"""Phase E (part 1): draw boxes, labels and confidences on a copy of the image."""
import cv2

import config


def annotate(image, regions):
    """Return a NEW image with boxes drawn; the input array is never modified."""
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(round(min(h, w) / 300)))
    scale = max(0.5, min(h, w) / 900)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        color = config.COLORS[r["label"]]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
        text = f'{r["label"]} {r["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness - 1))
        ty = y0 - 4 if y0 - th - base - 4 >= 0 else y0 + th + base + 4
        tx = min(x0, max(0, w - tw - 2))
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 4, ty + base), color, -1)
        cv2.putText(out, text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return out
