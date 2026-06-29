"""Tests for the LBPH recognizer and score-mapping logic.

The deep (SFace) engine needs downloaded weights, so it is exercised only for
its pure score-mapping behaviour and otherwise skipped when unavailable.
"""

import os

import cv2
import numpy as np
import pytest

from core.detectors import Detection
from core.recognizers import LBPHRecognizer, SFaceRecognizer

pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "face"),
    reason="opencv-contrib (cv2.face) not available",
)


def _striped(orientation: str, seed: int) -> np.ndarray:
    """Make a 100×100 grayscale image with strong directional texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((100, 100), dtype=np.uint8)
    if orientation == "vertical":
        img[:, ::4] = 255
    else:
        img[::4, :] = 255
    noise = rng.integers(0, 30, size=img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _make_dataset(root: str):
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    samples = []
    for i in range(12):
        v = _striped("vertical", i)
        h = _striped("horizontal", 100 + i)
        pv = os.path.join(root, "dataset", f"User.100.{i}.jpg")
        ph = os.path.join(root, "dataset", f"User.200.{i}.jpg")
        cv2.imwrite(pv, v)
        cv2.imwrite(ph, h)
        samples.append((pv, 100))
        samples.append((ph, 200))
    return samples


def test_lbph_trains_and_classifies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples = _make_dataset(str(tmp_path))

    rec = LBPHRecognizer()
    n = rec.train(samples)
    assert n == 2
    assert rec.model_exists()

    # held-out vertical image should classify as identity 100
    probe_gray = _striped("vertical", 999)
    frame = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
    det = Detection(0, 0, 100, 100)
    student_id, distance = rec.predict(frame, det)
    assert student_id == 100
    assert distance >= 0


def test_lbph_augmented_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples = _make_dataset(str(tmp_path))
    rec = LBPHRecognizer()
    assert rec.train(samples, augment=True) == 2


def test_lbph_score_mapping():
    rec = LBPHRecognizer()
    # lower distance = better; is_match compares distance < threshold
    assert rec.is_match(30, threshold=80) is True
    assert rec.is_match(90, threshold=80) is False
    # confidence is clamped to [0, 100]
    assert rec.confidence_pct(0) == 100
    assert rec.confidence_pct(150) == 0


def test_sface_score_mapping():
    rec = SFaceRecognizer()
    # cosine similarity: higher = better; threshold stored as percentage
    assert rec.is_match(0.5, threshold=36) is True
    assert rec.is_match(0.2, threshold=36) is False
    assert rec.confidence_pct(0.9) == 90
    assert rec.confidence_pct(-1) == 0
