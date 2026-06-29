#!/usr/bin/env python3
"""Offline model evaluation — the rigorous, reproducible benchmark.

This is what separates a working demo from an engineered ML project: instead
of claiming an accuracy number, this script *measures* one. It performs a
stratified train/test split per student, trains the chosen recognition engine
on the training split, predicts the held-out test split, and reports:

    • Overall accuracy
    • Precision / recall / F1 (macro-averaged + per-class)
    • A confusion matrix, saved as a PNG to assets/

Usage:
    python evaluate.py                  # classical engine (Haar + LBPH)
    python evaluate.py --engine deep    # deep engine (YuNet + SFace)
    python evaluate.py --test-size 0.3 --augment

Run it on your own dataset/ to generate the numbers and figures cited in the
README.
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from collections import defaultdict

# Windows consoles default to cp1252 and crash on box-drawing chars — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

import core
from config import (
    FACE_SIZE, SFACE_ALIGN_SIZE, ASSETS_DIR,
    SFACE_THRESHOLD, LBPH_THRESHOLD,
)
from core import dataset as ds
from core.augmentation import augment_samples

log = logging.getLogger("evaluate")


# ── splitting ──────────────────────────────────────────────────────────────────

def stratified_split(samples, test_size: float, seed: int = 42):
    """Per-identity train/test split so every student appears in both sets."""
    rng = np.random.default_rng(seed)
    by_id: dict[int, list[str]] = defaultdict(list)
    for path, label in samples:
        by_id[label].append(path)

    train, test = [], []
    for label, paths in by_id.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_test = max(1, int(round(len(paths) * test_size))) if len(paths) > 1 else 0
        for p in paths[:n_test]:
            test.append((p, label))
        for p in paths[n_test:]:
            train.append((p, label))
    return train, test


# ── engine-specific train / predict on file lists ──────────────────────────────

def _eval_classical(train, test, augment):
    faces = [ds.load_gray(p, FACE_SIZE) for p, _ in train]
    ids = [lbl for _, lbl in train]
    if augment:
        faces, ids = augment_samples(faces, ids)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(ids))

    y_true, y_pred = [], []
    for path, label in test:
        crop = ds.load_gray(path, FACE_SIZE)
        pred, dist = model.predict(crop)
        y_true.append(label)
        y_pred.append(pred if dist < LBPH_THRESHOLD else -1)
    return y_true, y_pred


def _eval_deep(train, test, augment):
    from core.recognizers import SFaceRecognizer
    if not SFaceRecognizer.is_available():
        raise SystemExit(
            "Deep engine unavailable — run  python download_models.py  first.")

    rec = SFaceRecognizer()
    sf = rec._engine()  # noqa: SLF001 — evaluation needs the raw embedder

    def embed(path):
        img = ds.load_bgr(path, SFACE_ALIGN_SIZE)
        v = sf.feature(img).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    # build gallery from the training split
    per_id = defaultdict(list)
    for path, label in train:
        per_id[label].append(embed(path))
    ids = sorted(per_id)
    gallery = np.stack([
        (lambda m: m / (np.linalg.norm(m) + 1e-10))(np.mean(np.stack(per_id[i]), axis=0))
        for i in ids
    ])
    ids_arr = np.array(ids)

    thr = SFACE_THRESHOLD / 100.0
    y_true, y_pred = [], []
    for path, label in test:
        v = embed(path)
        cos = gallery @ v
        best = int(np.argmax(cos))
        y_true.append(label)
        y_pred.append(int(ids_arr[best]) if cos[best] >= thr else -1)
    return y_true, y_pred


# ── metrics ─────────────────────────────────────────────────────────────────────

def _manual_report(y_true, y_pred, labels):
    """Precision/recall/F1 without sklearn (fallback)."""
    acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    lines = [f"Accuracy: {acc:.3f}\n",
             f"{'class':>8} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}"]
    precs, recs, f1s = [], [], []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        lines.append(f"{c:>8} {prec:>7.2f} {rec:>7.2f} {f1:>7.2f} {support:>8}")
    lines.append(f"\n{'macro':>8} {np.mean(precs):>7.2f} "
                 f"{np.mean(recs):>7.2f} {np.mean(f1s):>7.2f}")
    return acc, "\n".join(lines)


def _confusion_png(y_true, y_pred, labels, engine, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping confusion-matrix plot")
        return None

    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p in idx:                  # ignore "rejected as unknown" (-1)
            cm[idx[t], idx[p]] += 1

    fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {engine} engine")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"confusion_matrix_{engine}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    core.configure_logging()
    ap = argparse.ArgumentParser(description="Evaluate a recognition engine.")
    ap.add_argument("--engine", choices=["classical", "deep"], default="classical")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--augment", action="store_true",
                    help="augment the training split (classical only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = ds.list_samples()
    n_ids = len({lbl for _, lbl in samples})
    if n_ids < 2:
        print("\nNeed at least 2 enrolled students with images to evaluate.")
        print("Register students in the app, then re-run.\n")
        return 1

    train, test = stratified_split(samples, args.test_size, args.seed)
    print(f"\n  Engine     : {args.engine}")
    print(f"  Identities : {n_ids}")
    print(f"  Train / Test: {len(train)} / {len(test)} images")
    print("─" * 52)

    if args.engine == "classical":
        y_true, y_pred = _eval_classical(train, test, args.augment)
    else:
        y_true, y_pred = _eval_deep(train, test, args.augment)

    labels = sorted({lbl for _, lbl in samples})

    # prefer sklearn for a polished report, fall back to a manual one
    try:
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
    except ImportError:
        acc, report = _manual_report(y_true, y_pred, labels)

    print(report)
    print("─" * 52)
    print(f"  Overall accuracy: {acc:.1%}")

    png = _confusion_png(y_true, y_pred, labels, args.engine, ASSETS_DIR)
    if png:
        print(f"  Confusion matrix saved → {png}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
