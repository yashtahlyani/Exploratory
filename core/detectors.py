"""Face detection backends.

Two interchangeable detectors implement the same :class:`BaseDetector`
interface so the rest of the system never cares which one is active:

    HaarDetector  — Viola–Jones cascade (2001). Fast, CPU-only, ships with
                    OpenCV. Sensitive to pose and lighting.
    YuNetDetector — A lightweight CNN (2023) from the OpenCV Zoo. Far more
                    robust, returns 5 facial landmarks used for alignment,
                    and still runs comfortably on CPU.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    MIN_FACE_PX, YUNET_MODEL, YUNET_SCORE_THRESHOLD, CASCADE_PATH_RELATIVE,
)

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected face in image coordinates.

    ``raw`` holds the 15-column YuNet row (bbox + 5 landmarks + score) when
    available, which SFace uses for geometric alignment. It is ``None`` for
    the Haar detector.
    """
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    raw: np.ndarray | None = field(default=None, repr=False)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR frame and return them in image coordinates."""

    @staticmethod
    def is_available() -> bool:
        return True


class HaarDetector(BaseDetector):
    name = "haar"

    def __init__(self):
        cascade_path = cv2.data.haarcascades + CASCADE_PATH_RELATIVE
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            minSize=(MIN_FACE_PX, MIN_FACE_PX),
        )
        return [Detection(int(x), int(y), int(w), int(h))
                for (x, y, w, h) in boxes]


class YuNetDetector(BaseDetector):
    name = "yunet"

    def __init__(self, model_path: str = YUNET_MODEL,
                 score_threshold: float = YUNET_SCORE_THRESHOLD):
        if not self.is_available(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. "
                "Run  python download_models.py  first."
            )
        # input size is updated per-frame in detect()
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold, 0.3, 5000,
        )
        self._input_size: tuple[int, int] | None = None

    @staticmethod
    def is_available(model_path: str = YUNET_MODEL) -> bool:
        import os
        return hasattr(cv2, "FaceDetectorYN") and os.path.exists(model_path)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        detections: list[Detection] = []
        for row in faces:
            x, y, bw, bh = row[:4]
            if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
                continue
            detections.append(
                Detection(
                    x=max(int(x), 0), y=max(int(y), 0),
                    w=int(bw), h=int(bh),
                    score=float(row[-1]),
                    raw=row.astype(np.float32),
                )
            )
        return detections
