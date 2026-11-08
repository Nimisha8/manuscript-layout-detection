"""All tunable settings in one place."""

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

CLASSES = ["header", "footer", "main_text", "side_text", "filler"]

# BGR colours used for annotation
COLORS = {
    "header": (0, 140, 255),     # orange
    "footer": (255, 0, 255),     # magenta
    "main_text": (0, 170, 0),    # green
    "side_text": (255, 120, 0),  # blue
    "filler": (0, 0, 220),       # red
}

# --- preprocessing ---
MAX_SIDE = 2400               # larger images are analysed at this size, boxes scaled back
CLAHE_CLIP = 2.0
THRESH_C = 12
DARK_PIXEL_MAX = 50              # off-page text must be this dark (0-255)
STRONG_INK_RATIO = 0.55       # pixel darker than 55% of local background = definite ink                 # adaptive threshold offset (lower = keeps fainter ink)
MAX_SKEW_DEG = 6.0
MIN_SKEW_TO_CORRECT = 0.5

# --- detection (sizes are relative to the typical character height "unit") ---
MIN_COMPONENT_AREA_FRAC = 0.00001   # of image area
LINE_DILATE_X = 2.5                 # horizontal grouping distance, in units
LINE_DILATE_Y = 0.2
BLOCK_MAX_VGAP = 1.5                # max vertical gap (in line heights) to merge lines
MAX_CHAR_SIZE_RATIO = 1.8           # lines with very different letter sizes are not merged
BLOCK_MIN_HOVERLAP = 0.3
COLUMN_GAP_UNITS = 2.0              # min empty vertical band to split columns
BOX_PAD_CHAR_FRAC = 0.15            # box padding as a fraction of character height
TRIM_PERCENTILE = 0.6               # box edges ignore this % of outermost ink pixels
MIN_BLOCK_INK_FRAC = 0.00002

# --- classification ---
SMALL_TEXT_RATIO = 0.65             # char height vs main text below this => "small print"
THIN_STROKE_RATIO = 0.85            # pen stroke vs main text; printed fonts are thinner
MIN_REL_INK = 0.01                  # blocks with less ink than this (vs main body) are dropped
FAINT_CONTRAST_RATIO = 0.45         # ink contrast vs main text below this => pencil/faint
SIDE_MAX_WIDTH_RATIO = 0.45         # side text width vs main text width
