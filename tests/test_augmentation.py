"""Tests for the data-augmentation utilities."""

import numpy as np

from core import augmentation as aug


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(100, 100), dtype=np.uint8)


def test_augment_image_count_shape_dtype():
    img = _sample()
    variants = aug.augment_image(img, rng=np.random.default_rng(1))
    assert len(variants) == 4
    for v in variants:
        assert v.shape == img.shape
        assert v.dtype == np.uint8


def test_flip_is_reversible():
    img = _sample()
    assert np.array_equal(aug.flip_horizontal(aug.flip_horizontal(img)), img)


def test_brightness_bounds():
    img = _sample()
    bright = aug.adjust_brightness(img, 5.0)   # heavy over-exposure
    dark = aug.adjust_brightness(img, 0.0)
    assert bright.max() <= 255 and bright.min() >= 0
    assert np.all(dark == 0)


def test_augment_samples_expands_and_aligns_labels():
    faces = [_sample(i) for i in range(3)]
    ids = [10, 20, 30]
    out_faces, out_ids = aug.augment_samples(faces, ids,
                                             rng=np.random.default_rng(2))
    # original + 4 variants each
    assert len(out_faces) == len(faces) * 5
    assert len(out_ids) == len(faces) * 5
    # first element of each block is the original label, in order
    assert out_ids[0] == 10 and out_ids[5] == 20 and out_ids[10] == 30
