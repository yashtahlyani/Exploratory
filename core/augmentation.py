"""Training-time data augmentation.

Capturing 50 images of a face still leaves the model brittle to lighting,
pose, and sensor noise it never saw during enrollment. Augmentation expands
each sample into several plausible variations, improving generalization
without collecting more data — a standard technique for small-data vision.

All functions operate on grayscale ``uint8`` arrays and return arrays of the
same shape and dtype, so augmented samples are drop-in for the recognizers.
"""

import cv2
import numpy as np

# Deterministic-by-default RNG; pass a seed in tests for reproducibility.
_rng = np.random.default_rng()


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Mirror left↔right — faces are roughly symmetric, so this is label-safe."""
    return cv2.flip(img, 1)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensities to simulate brighter / darker rooms."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    """Small in-plane rotation to simulate a slightly tilted head."""
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    """Add sensor-like noise to improve robustness to grainy webcams."""
    rng = rng or _rng
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_image(img: np.ndarray, rng=None) -> list[np.ndarray]:
    """Return a list of augmented variants of ``img`` (excluding the original).

    Produces a fixed, curated set rather than random sampling so that the
    augmented dataset size is predictable: 4 variants per source image.
    """
    rng = rng or _rng
    return [
        flip_horizontal(img),
        adjust_brightness(img, 1.25),          # brighter
        adjust_brightness(img, 0.75),          # darker
        rotate(img, float(rng.uniform(-12, 12))),
    ]


def augment_samples(
    faces: list[np.ndarray],
    ids: list[int],
    rng=None,
) -> tuple[list[np.ndarray], list[int]]:
    """Expand a labelled dataset in-memory with augmented variants.

    Returns ``(faces_out, ids_out)`` containing the originals followed by
    their augmentations, with labels kept aligned.
    """
    faces_out: list[np.ndarray] = []
    ids_out: list[int] = []
    for face, label in zip(faces, ids):
        faces_out.append(face)
        ids_out.append(label)
        for variant in augment_image(face, rng=rng):
            faces_out.append(variant)
            ids_out.append(label)
    return faces_out, ids_out
