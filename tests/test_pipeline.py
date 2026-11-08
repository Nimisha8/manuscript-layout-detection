import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import inference  # noqa: E402
from src import preprocessing, region_detector, visualization  # noqa: E402
from src.utils import clip_box, is_valid_region  # noqa: E402

FONT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX


def make_page(skew=0.0, faded=False, noise=False):
    """Synthetic paper page: header, 8 body lines, footer and a margin note."""
    img = np.full((1000, 800, 3), 235, np.uint8)
    ink = 120 if faded else 20
    def text(s, x, y, scale):
        cv2.putText(img, s, (x, y), FONT, scale, (ink, ink, ink), 3, cv2.LINE_AA)
    text("Chapter One  12", 300, 90, 1.4)
    for i in range(8):
        text("manuscript body line %d" % i, 220, 230 + i * 85, 1.7)
    text("catchword", 300, 940, 1.4)
    for i in range(4):
        text("note %d" % i, 20, 300 + i * 40, 0.9)  # left margin note
    if noise:
        img = cv2.add(img, np.random.RandomState(0).randint(0, 25, img.shape, dtype=np.uint8))
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if skew:
        m = cv2.getRotationMatrix2D((400, 500), skew, 1.0)
        img = cv2.warpAffine(img, m, (800, 1000), borderValue=(235, 235, 235))
    return img


@pytest.fixture
def page():
    return make_page()


def test_load_image_and_corrupted(tmp_path):
    good = tmp_path / "a.png"
    cv2.imwrite(str(good), make_page())
    assert preprocessing.load_image(good).shape == (1000, 800, 3)
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        preprocessing.load_image(bad)


def test_preprocessing_binary(page):
    pre = preprocessing.preprocess(page)
    assert pre["binary"].shape == page.shape[:2]
    assert set(np.unique(pre["binary"])) <= {0, 255}
    assert 0.01 < (pre["binary"] > 0).mean() < 0.5


def test_region_detection_finds_blocks(page):
    blocks, _ = region_detector.detect_blocks(preprocessing.preprocess(page))
    assert len(blocks) >= 3


def test_all_expected_labels_on_synthetic_page(page):
    regions, _ = inference.analyze_image(page)
    labels = {r["label"] for r in regions}
    assert {"header", "footer", "main_text", "side_text"} <= labels


def test_label_positions(page):
    regions, _ = inference.analyze_image(page)
    main = [r for r in regions if r["label"] == "main_text"][0]["bbox"]
    for r in regions:
        if r["label"] == "header":
            assert r["bbox"][3] <= main[1] + 10
        if r["label"] == "footer":
            assert r["bbox"][1] >= main[3] - 10


@pytest.mark.parametrize("kwargs", [dict(), dict(skew=3), dict(faded=True), dict(noise=True)])
def test_boxes_inside_image_and_valid(kwargs):
    img = make_page(**kwargs)
    h, w = img.shape[:2]
    regions, _ = inference.analyze_image(img)
    assert regions
    for r in regions:
        assert is_valid_region(r, w, h)
        assert r["label"] in config.CLASSES
        assert 0 <= r["confidence"] <= 1


def test_clip_box():
    assert clip_box([-5, -5, 2000, 2000], 100, 50) == [0, 0, 100, 50]
    assert clip_box([10, 10, 10, 20], 100, 50) is None


def test_annotation_does_not_modify_input(page):
    before = page.copy()
    out = visualization.annotate(page, [{"bbox": [10, 10, 100, 100], "label": "header", "confidence": 0.9}])
    assert np.array_equal(page, before)
    assert not np.array_equal(out, page)


def test_json_output_and_original_preserved(tmp_path):
    src = tmp_path / "page.png"
    cv2.imwrite(str(src), make_page())
    original_bytes = src.read_bytes()
    out = tmp_path / "out"
    inference.process_file(src, out)
    data = json.loads((out / "predictions" / "page.json").read_text())
    assert data["filename"] == "page.png"
    assert data["image_width"] == 800 and data["image_height"] == 1000
    assert data["regions"] and {"bbox", "label", "confidence"} <= set(data["regions"][0])
    assert (out / "annotated" / "page_annotated.png").exists()
    assert src.read_bytes() == original_bytes


def test_batch_with_corrupted_file(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    cv2.imwrite(str(inp / "one.png"), make_page())
    cv2.imwrite(str(inp / "two.jpg"), make_page(skew=2))
    (inp / "broken.png").write_text("garbage")
    out = tmp_path / "out"
    code = inference.main(["--input", str(inp), "--output", str(out)])
    assert code == 0
    assert len(list((out / "predictions").glob("*.json"))) == 2
    assert "1 failed" in capsys.readouterr().out


def test_cli_bad_path(tmp_path):
    assert inference.main(["--input", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 2
