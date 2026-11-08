"""Command line entry point.

    python inference.py --input ./data/input --output ./results
    python inference.py --input ./data/input/page.png --output ./results
"""
import argparse
import sys
from pathlib import Path

import config
from src import preprocessing, region_detector, region_classifier, visualization
from src.utils import list_images, is_valid_region, save_image, save_json


def analyze_image(image):
    """Run detection + classification on a BGR image. Returns (regions, skew_angle)."""
    h, w = image.shape[:2]
    pre = preprocessing.preprocess(image)
    blocks, angle = region_detector.detect_blocks(pre)
    regions = region_classifier.classify_blocks(blocks, w, h)
    regions = [r for r in regions if is_valid_region(r, w, h)]
    return regions, angle


def process_file(image_path, output_dir):
    """Process one image file and write annotated image + JSON. Raises on bad files."""
    image_path, output_dir = Path(image_path), Path(output_dir)
    image = preprocessing.load_image(image_path)
    h, w = image.shape[:2]
    regions, angle = analyze_image(image)

    annotated = visualization.annotate(image, regions)
    save_image(output_dir / "annotated" / f"{image_path.stem}_annotated.png", annotated)
    result = {
        "filename": image_path.name,
        "image_width": w,
        "image_height": h,
        "skew_angle_deg": round(angle, 2),
        "num_regions": len(regions),
        "regions": [{"bbox": r["bbox"], "label": r["label"], "confidence": r["confidence"]}
                    for r in regions],
    }
    save_json(output_dir / "predictions" / f"{image_path.stem}.json", result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manuscript layout region detection")
    p.add_argument("--input", required=True, help="image file or folder of images")
    p.add_argument("--output", default="./results", help="output folder (default ./results)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        print(f"Error: input path not found: {src}", file=sys.stderr)
        return 2
    if src.is_file():
        if src.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            print(f"Error: unsupported file type: {src.suffix}", file=sys.stderr)
            return 2
        files = [src]
    else:
        files = list_images(src)
        if not files:
            print(f"Error: no supported images found in {src}", file=sys.stderr)
            return 2

    out = Path(args.output)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name} ... ", end="", flush=True)
        try:
            res = process_file(f, out)
            counts = {}
            for r in res["regions"]:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            print("ok", counts)
            ok.append(f.name)
        except Exception as e:  # corrupted or unreadable image: report and carry on
            print(f"FAILED ({e})")
            failed.append((f.name, str(e)))

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed. Output: {out}")
    for name, err in failed:
        print(f"  failed: {name}: {err}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
