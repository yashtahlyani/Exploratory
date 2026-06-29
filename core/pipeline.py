"""Recognition pipeline — the single object the GUI and evaluator talk to.

An *engine* couples a detector with a recognizer:

    "classical"  →  Haar Cascade  +  LBPH
    "deep"       →  YuNet CNN     +  SFace embeddings

The pipeline hides every engine-specific detail (score orientation, default
threshold, model files) behind one uniform API, and degrades gracefully: if
the deep-learning weights have not been downloaded, only the classical engine
is offered, so the app always runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import (
    DEFAULT_ENGINE,
    LBPH_THRESHOLD, LBPH_THRESHOLD_RANGE,
    SFACE_THRESHOLD, SFACE_THRESHOLD_RANGE,
)
from core import dataset as ds
from core.detectors import BaseDetector, HaarDetector, YuNetDetector, Detection
from core.recognizers import BaseRecognizer, LBPHRecognizer, SFaceRecognizer

log = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Outcome of recognizing one detected face."""
    student_id: int
    recognized: bool
    confidence_pct: int
    raw_score: float


@dataclass
class _EngineSpec:
    key: str
    label: str
    detector_cls: type[BaseDetector]
    recognizer_cls: type[BaseRecognizer]
    default_threshold: int
    threshold_range: tuple[int, int]
    threshold_caption: str


_ENGINES: dict[str, _EngineSpec] = {
    "classical": _EngineSpec(
        key="classical",
        label="Classical  (Haar + LBPH)",
        detector_cls=HaarDetector,
        recognizer_cls=LBPHRecognizer,
        default_threshold=LBPH_THRESHOLD,
        threshold_range=LBPH_THRESHOLD_RANGE,
        threshold_caption="LBPH distance — lower = stricter",
    ),
    "deep": _EngineSpec(
        key="deep",
        label="Deep Learning  (YuNet + SFace)",
        detector_cls=YuNetDetector,
        recognizer_cls=SFaceRecognizer,
        default_threshold=SFACE_THRESHOLD,
        threshold_range=SFACE_THRESHOLD_RANGE,
        threshold_caption="cosine % — higher = stricter",
    ),
}


def available_engines() -> list[str]:
    """Engine keys whose detector + recognizer dependencies are present."""
    keys = ["classical"]   # always available with opencv-contrib
    if YuNetDetector.is_available() and SFaceRecognizer.is_available():
        keys.append("deep")
    return keys


def engine_label(key: str) -> str:
    return _ENGINES[key].label


class RecognitionPipeline:
    def __init__(self, engine: str = DEFAULT_ENGINE):
        if engine not in available_engines():
            engine = "classical"
        self._engine_key = ""
        self._detector: BaseDetector | None = None
        self._recognizer: BaseRecognizer | None = None
        self.threshold: float = 0
        self.set_engine(engine)

    # ── engine management ──────────────────────────────────────────────────────

    @property
    def engine_key(self) -> str:
        return self._engine_key

    @property
    def engine_label(self) -> str:
        return _ENGINES[self._engine_key].label

    def set_engine(self, key: str) -> bool:
        """Switch engine. Returns False (and keeps current) if unavailable."""
        if key not in available_engines():
            log.warning("Engine '%s' unavailable; keeping '%s'",
                        key, self._engine_key or "none")
            return False
        spec = _ENGINES[key]
        self._detector = spec.detector_cls()
        self._recognizer = spec.recognizer_cls()
        self._engine_key = key
        self.threshold = spec.default_threshold
        log.info("Engine set to '%s'", key)
        return True

    def threshold_range(self) -> tuple[int, int]:
        return _ENGINES[self._engine_key].threshold_range

    def threshold_caption(self) -> str:
        return _ENGINES[self._engine_key].threshold_caption

    # ── training ───────────────────────────────────────────────────────────────

    def model_exists(self) -> bool:
        return self._recognizer.model_exists()

    def train(self, augment: bool = False) -> int:
        samples = ds.list_samples()
        return self._recognizer.train(samples, augment=augment)

    # ── inference ──────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return self._detector.detect(frame_bgr)

    def recognize(self, frame_bgr: np.ndarray, det: Detection) -> RecognitionResult:
        student_id, raw = self._recognizer.predict(frame_bgr, det)
        recognized = (student_id >= 0
                      and self._recognizer.is_match(raw, self.threshold))
        return RecognitionResult(
            student_id=student_id,
            recognized=recognized,
            confidence_pct=self._recognizer.confidence_pct(raw),
            raw_score=raw,
        )
