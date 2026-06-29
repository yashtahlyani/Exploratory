"""Backward-compatibility shim.

The training/recognition logic now lives in the :mod:`core` package, which
powers both the GUI and the evaluation scripts. This module is kept so that
older imports (``from face_trainer import FaceTrainer``) keep working, but it
simply delegates to :class:`core.recognizers.LBPHRecognizer`.

New code should use :class:`core.pipeline.RecognitionPipeline` instead.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import FACE_SIZE, MODEL_PATH, TRAINER_DIR, DATASET_DIR
from core import dataset as ds
from core.recognizers import LBPHRecognizer


class FaceTrainer:
    DATASET_DIR = DATASET_DIR
    TRAINER_DIR = TRAINER_DIR
    MODEL_PATH = MODEL_PATH

    def __init__(self):
        self._rec = LBPHRecognizer()

    def train(self, augment: bool = False) -> int:
        """Train the LBPH model; returns the number of unique students."""
        return self._rec.train(ds.list_samples(), augment=augment)

    def model_exists(self) -> bool:
        return self._rec.model_exists()

    def load_recognizer(self):
        """Return a raw, loaded ``cv2`` LBPH recognizer (legacy callers)."""
        self._rec.load()
        return self._rec._model            # noqa: SLF001 — intentional legacy access

    @staticmethod
    def prepare_face(gray_roi: np.ndarray) -> np.ndarray:
        """Resize a grayscale face ROI to the training size."""
        return cv2.resize(gray_roi, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
