# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [3.0.0] — 2026-06-30

### Added — Deep learning, evaluation, and engineering rigour
- **Dual recognition engines** behind a clean `core/` abstraction, switchable
  live from the toolbar:
  - *Classical* — Haar Cascade detector + LBPH recognizer (default, no downloads)
  - *Deep Learning* — YuNet CNN detector + SFace 128-D embeddings with cosine
    similarity (the FaceNet/ArcFace paradigm, CPU-only)
- `evaluate.py` — reproducible benchmark: stratified train/test split, accuracy,
  precision/recall/F1, and a saved confusion matrix.
- `core/augmentation.py` — training-time augmentation (flip, brightness, rotation).
- `download_models.py` — fetches YuNet + SFace weights from the OpenCV Zoo.
- `tests/` — pytest suite (augmentation, attendance manager, recognizers, pipeline).
- GitHub Actions CI (`.github/workflows/ci.yml`) — lint + tests on every push.
- `notebooks/model_evaluation.ipynb` — interactive engine comparison.
- Structured logging to console and `attendance.log`.

### Changed
- `app.py` now routes all detection/recognition/training through
  `RecognitionPipeline` and no longer calls OpenCV directly.
- `face_trainer.py` reduced to a thin backward-compatibility shim over `core`.
- `config.py` extended with engine settings, per-engine thresholds, and model paths.

### Fixed
- Windows console (cp1252) crash when CLI scripts printed `✓`/`✗`/box glyphs —
  scripts now force UTF-8 stdout.

## [2.0.0] — 2026-06-30

### Added
- Analytics dashboard (matplotlib): daily trend, per-student attendance %, summary.
- Student Registry: view enrolled students with image counts, delete with cleanup.
- Live confidence-threshold slider; audio beep and live "marked today" counter.
- Records search/filter by name; per-session and bulk Excel export.
- `config.py` central constants and `check_env.py` pre-flight checker.

## [1.0.0] — 2026-06-29

### Added
- Initial face-recognition attendance system: real-time detection + LBPH
  recognition, 50-image enrollment, daily CSV records, dark Tkinter UI.
- Project README, MIT license.

[3.0.0]: https://github.com/yashtahlyani/Exploratory/releases/tag/v3.0.0
[2.0.0]: https://github.com/yashtahlyani/Exploratory/releases/tag/v2.0.0
[1.0.0]: https://github.com/yashtahlyani/Exploratory/releases/tag/v1.0.0
