"""Face recognition backends.

Both implement the same :class:`BaseRecognizer` interface, so the pipeline can
swap them freely:

    LBPHRecognizer  — Local Binary Pattern Histograms (classical). Encodes
                      local texture; compares with a Chi-square *distance*
                      (lower = better). Trains in milliseconds, no weights.

    SFaceRecognizer — A CNN (OpenCV Zoo, 2021) trained with a margin loss that
                      maps each face to a 128-D unit embedding. Identity is
                      decided by *cosine similarity* (higher = better) against
                      a per-student gallery embedding. This is the same
                      paradigm as FaceNet / ArcFace, just small enough for CPU.

Native scores differ (distance vs. similarity), so each recognizer also knows
how to turn its raw score into a 0–100 "confidence %" and how to decide a
match given a threshold — letting the GUI drive both with one slider.
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from config import (
    FACE_SIZE, SFACE_ALIGN_SIZE, TRAINER_DIR,
    MODEL_PATH, SFACE_MODEL, SFACE_GALLERY,
)
from core import dataset as ds
from core.augmentation import augment_samples
from core.detectors import Detection

log = logging.getLogger(__name__)


class BaseRecognizer(ABC):
    name: str = "base"
    higher_is_better: bool = True   # orientation of the raw score

    @abstractmethod
    def train(self, samples: list[tuple[str, int]], augment: bool = False) -> int:
        """Train on ``[(path, id), ...]`` and persist. Returns #identities."""

    @abstractmethod
    def load(self) -> None:
        """Load a previously trained model from disk."""

    @abstractmethod
    def model_exists(self) -> bool:
        """Whether a trained model is present on disk."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray, det: Detection) -> tuple[int, float]:
        """Return ``(student_id, raw_score)`` for one detected face."""

    @abstractmethod
    def is_match(self, raw_score: float, threshold: float) -> bool:
        """Decide recognition given the engine-native ``threshold``."""

    @abstractmethod
    def confidence_pct(self, raw_score: float) -> int:
        """Map a raw score to a 0–100 confidence for display."""

    @staticmethod
    def is_available() -> bool:
        return True


# ───────────────────────────────────────────────────────────── LBPH (classical)

class LBPHRecognizer(BaseRecognizer):
    name = "lbph"
    higher_is_better = False   # LBPH distance: lower = better

    def __init__(self):
        self._model = None

    @staticmethod
    def is_available() -> bool:
        return hasattr(cv2, "face")

    def _new_model(self):
        return cv2.face.LBPHFaceRecognizer_create()

    def train(self, samples: list[tuple[str, int]], augment: bool = False) -> int:
        if not samples:
            raise ValueError("No face images found in the dataset folder.")

        faces = [ds.load_gray(p, FACE_SIZE) for p, _ in samples]
        ids = [label for _, label in samples]

        if augment:
            faces, ids = augment_samples(faces, ids)
            log.info("LBPH training set expanded to %d samples via augmentation",
                     len(faces))

        os.makedirs(TRAINER_DIR, exist_ok=True)
        model = self._new_model()
        model.train(faces, np.array(ids))
        model.write(MODEL_PATH)
        self._model = model
        n = len(set(ids))
        log.info("LBPH model trained on %d identities, %d samples", n, len(faces))
        return n

    def load(self) -> None:
        if not self.model_exists():
            raise FileNotFoundError(f"No LBPH model at {MODEL_PATH}")
        model = self._new_model()
        model.read(MODEL_PATH)
        self._model = model

    def model_exists(self) -> bool:
        return os.path.exists(MODEL_PATH)

    def predict(self, frame_bgr: np.ndarray, det: Detection) -> tuple[int, float]:
        if self._model is None:
            self.load()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x, y, w, h = det.box
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            return -1, 1e9
        crop = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_LANCZOS4)
        label, distance = self._model.predict(crop)
        return int(label), float(distance)

    def is_match(self, raw_score: float, threshold: float) -> bool:
        return raw_score < threshold

    def confidence_pct(self, raw_score: float) -> int:
        return int(max(0, min(100, round(100 - raw_score))))


# ───────────────────────────────────────────────────────────── SFace (deep CNN)

class SFaceRecognizer(BaseRecognizer):
    name = "sface"
    higher_is_better = True   # cosine similarity: higher = better

    def __init__(self, model_path: str = SFACE_MODEL):
        self._model_path = model_path
        self._sf = None                       # cv2.FaceRecognizerSF
        self._ids: np.ndarray | None = None   # gallery labels
        self._embeds: np.ndarray | None = None  # gallery embeddings (N, 128) L2-normed

    @staticmethod
    def is_available() -> bool:
        return hasattr(cv2, "FaceRecognizerSF") and os.path.exists(SFACE_MODEL)

    def _engine(self):
        if self._sf is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"SFace model not found at {self._model_path}. "
                    "Run  python download_models.py  first."
                )
            self._sf = cv2.FaceRecognizerSF.create(self._model_path, "")
        return self._sf

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-10
        return v / n

    def _embed(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        """Compute an L2-normalized 128-D embedding for one face."""
        sf = self._engine()
        if det.raw is not None:
            aligned = sf.alignCrop(frame_bgr, det.raw)
        else:
            x, y, w, h = det.box
            crop = frame_bgr[y:y + h, x:x + w]
            aligned = cv2.resize(crop, SFACE_ALIGN_SIZE,
                                 interpolation=cv2.INTER_LANCZOS4)
        feat = sf.feature(aligned).flatten().astype(np.float32)
        return self._l2norm(feat)

    def _embed_path(self, path: str) -> np.ndarray:
        """Embed a stored dataset crop (no landmarks → centre-resize)."""
        img = ds.load_bgr(path, SFACE_ALIGN_SIZE)
        feat = self._engine().feature(img).flatten().astype(np.float32)
        return self._l2norm(feat)

    def train(self, samples: list[tuple[str, int]], augment: bool = False) -> int:
        if not samples:
            raise ValueError("No face images found in the dataset folder.")

        per_id: dict[int, list[np.ndarray]] = {}
        for path, label in samples:
            try:
                per_id.setdefault(label, []).append(self._embed_path(path))
            except Exception as exc:                       # pragma: no cover
                log.warning("Skipping %s: %s", path, exc)

        ids, embeds = [], []
        for label, vecs in sorted(per_id.items()):
            mean = self._l2norm(np.mean(np.stack(vecs), axis=0))
            ids.append(label)
            embeds.append(mean)

        self._ids = np.array(ids, dtype=np.int32)
        self._embeds = np.stack(embeds).astype(np.float32)

        os.makedirs(TRAINER_DIR, exist_ok=True)
        np.savez(SFACE_GALLERY, ids=self._ids, embeds=self._embeds)
        log.info("SFace gallery built for %d identities", len(ids))
        return len(ids)

    def load(self) -> None:
        if not os.path.exists(SFACE_GALLERY):
            raise FileNotFoundError(f"No SFace gallery at {SFACE_GALLERY}")
        data = np.load(SFACE_GALLERY)
        self._ids = data["ids"]
        self._embeds = data["embeds"].astype(np.float32)

    def model_exists(self) -> bool:
        return os.path.exists(SFACE_GALLERY) and os.path.exists(self._model_path)

    def predict(self, frame_bgr: np.ndarray, det: Detection) -> tuple[int, float]:
        if self._embeds is None:
            self.load()
        if self._embeds is None or len(self._embeds) == 0:
            return -1, 0.0
        vec = self._embed(frame_bgr, det)
        cosines = self._embeds @ vec          # both L2-normalized → dot = cosine
        best = int(np.argmax(cosines))
        return int(self._ids[best]), float(cosines[best])

    def is_match(self, raw_score: float, threshold: float) -> bool:
        # threshold stored as a percentage (e.g. 36 → cosine 0.36)
        return raw_score >= (threshold / 100.0)

    def confidence_pct(self, raw_score: float) -> int:
        return int(max(0, min(100, round(raw_score * 100))))
