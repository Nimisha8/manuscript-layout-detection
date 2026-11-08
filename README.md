# Manuscript Layout Region Detection

Detects and classifies layout regions in historical manuscript images (palm-leaf and paper) into
`header`, `footer`, `main_text`, `side_text` and `filler`, and writes annotated images plus JSON predictions.

**This is a classical computer-vision pipeline (OpenCV + NumPy + rules). No deep-learning model is trained or used.**
No labelled data was available, so no accuracy figures are claimed (see Limitations).

## Problem and objectives
Given manuscript photos with different substrates, sizes, scripts and degradation, locate text regions and label
their layout role. Objectives: batch CLI, bounding boxes inside the image, confidence per region, JSON output,
originals never modified, modular readable code.

## Tech stack
Python 3, OpenCV, NumPy, Pillow (listed dependency), pytest.

## Architecture
```
inference.py            CLI + per-image pipeline (analyze_image, process_file)
config.py               all thresholds and colours
src/preprocessing.py    load, grayscale, background flattening, CLAHE, thresholding, line removal, page mask, skew
src/region_detector.py  components -> lines -> blocks -> column split -> block features
src/region_classifier.py rule-based labelling + confidence
src/visualization.py    annotated image
src/utils.py            file listing, box clipping/validation, saving
tests/test_pipeline.py  pytest suite (synthetic pages)
```

## Detection methodology
1. **Preprocessing** – grayscale; divide by a heavy blur (flattens uneven lighting/stains); denoise + CLAHE for faint ink;
   adaptive threshold with *hysteresis* (faint pixels are kept only if their stroke contains a definite dark pixel,
   which keeps faded ink but drops leaf/paper texture); long ruling/border lines removed by morphological opening.
   A page mask separates a leaf lying on a different background (e.g. cloth); off-page pixels count as ink only if near-black
   (printed watermark text). Skew is estimated from horizontal projection sharpness (±6°) and corrected for analysis;
   boxes are mapped back to the original image.
2. **Detection** – connected components (tiny specks and giant objects removed) -> horizontal morphological dilation
   groups characters into lines -> lines are stacked into blocks when vertically close, horizontally overlapping and of
   similar letter size / same page membership -> a vertical projection profile splits multi-column blocks ->
   box edges are trimmed of stray ink pixels.
3. **Classification** (relative to the *main body*, the block with most ink, so different page shapes work):
   - `filler`: text off the page, small thin-stroke print (e.g. English digitisation overlay), faint ink (pencil/stain)
   - `side_text`: narrow block beside the body overlapping its vertical extent
   - `header` / `footer`: block above / below the body, horizontally aligned with it
   - `main_text`: the body itself plus other substantial blocks overlapping its area
   Confidence is a rule-based score (alignment, overlap, size cues), not a calibrated probability.
4. **Validation** – boxes are `[x_min, y_min, x_max, y_max]` ints, clipped to the image, positive size, valid label, confidence in [0,1].

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python inference.py --input ./data/input --output ./results
python inference.py --input ./data/input/sample_paper.png --output ./results
pytest
```
Outputs: `results/annotated/<name>_annotated.png` and `results/predictions/<name>.json`.
Corrupted files are reported and skipped; a summary of successes/failures is printed.

## Output format
```json
{
  "filename": "sample_paper.png",
  "image_width": 2047,
  "image_height": 774,
  "skew_angle_deg": -0.5,
  "num_regions": 3,
  "regions": [{"bbox": [243, 94, 1982, 712], "label": "main_text", "confidence": 0.9}]
}
```


