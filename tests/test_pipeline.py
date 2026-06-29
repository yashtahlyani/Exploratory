"""Tests for engine selection and threshold handling in the pipeline."""

import cv2
import pytest

from core.pipeline import RecognitionPipeline, available_engines, engine_label

pytestmark = pytest.mark.skipif(
    not hasattr(cv2, "face"),
    reason="opencv-contrib (cv2.face) not available",
)


def test_classical_always_available():
    assert "classical" in available_engines()


def test_default_engine_is_classical():
    p = RecognitionPipeline()
    assert p.engine_key == "classical"


def test_threshold_within_declared_range():
    p = RecognitionPipeline()
    low, high = p.threshold_range()
    assert low <= p.threshold <= high
    assert p.threshold_caption()           # non-empty caption


def test_set_unavailable_engine_is_rejected():
    p = RecognitionPipeline()
    # 'deep' needs downloaded weights; in CI it is absent → switch must fail
    if "deep" not in available_engines():
        assert p.set_engine("deep") is False
        assert p.engine_key == "classical"


def test_engine_labels_exist_for_all_available():
    for key in available_engines():
        assert engine_label(key)
