"""Dataset loading helpers shared by the recognizers and the evaluator.

Images are stored as ``User.<id>.<n>.jpg`` grayscale face crops inside the
dataset directory. These helpers parse that convention into ``(path, id)``
pairs and load them as arrays so every consumer reads the data identically.
"""

import os
import logging

import cv2
import numpy as np

from config import DATASET_DIR

log = logging.getLogger(__name__)


def parse_label(filename: str) -> int | None:
    """Return the integer student id encoded in ``User.<id>.<n>.jpg``."""
    parts = os.path.basename(filename).split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def list_samples(dataset_dir: str = DATASET_DIR) -> list[tuple[str, int]]:
    """Return ``[(image_path, student_id), ...]`` for every labelled image."""
    if not os.path.isdir(dataset_dir):
        return []
    samples: list[tuple[str, int]] = []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        label = parse_label(fname)
        if label is None:
            continue
        samples.append((os.path.join(dataset_dir, fname), label))
    return samples


def load_gray(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as grayscale uint8, optionally resized."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def load_bgr(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as a 3-channel BGR uint8 array, optionally resized.

    Dataset crops are grayscale; SFace expects 3 channels, so a grayscale
    image is promoted to BGR by channel replication.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def count_identities(dataset_dir: str = DATASET_DIR) -> int:
    """Number of distinct student ids present in the dataset."""
    return len({label for _, label in list_samples(dataset_dir)})
