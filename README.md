<div align="center">

# Face Recognition Based Attendance System

**A dual-engine, contactless attendance platform — classical computer vision *and* deep-learning face recognition, benchmarked side by side.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Institute](https://img.shields.io/badge/IIT_(BHU)-Varanasi-B22222?style=for-the-badge)](https://iitbhu.ac.in)

[![CI](https://github.com/yashtahlyani/Exploratory/actions/workflows/ci.yml/badge.svg)](https://github.com/yashtahlyani/Exploratory/actions/workflows/ci.yml)

<br/>

*Exploratory Project · Department of Computer Science & Engineering*
*Indian Institute of Technology (BHU) Varanasi*

</div>

---

## The Problem

Traditional classroom attendance is slow, error-prone, and breaks down at scale. A roll-call for 60 students burns 5–8 minutes of class time, proxy attendance is trivially easy, and paper sheets need manual digitisation. This project replaces all of it: a single webcam marks every student automatically the moment they appear — no buttons, no paper, no manual work.

---

## What Makes This More Than a Demo

Most student face-recognition projects stop at "it detects my face." This one is built like a real ML system:

- **Two interchangeable recognition engines** behind one clean interface — *classical* (Haar + LBPH) and *deep learning* (YuNet + SFace CNN embeddings). Switch live from a dropdown.
- **A reproducible evaluation harness** (`evaluate.py`) that reports accuracy, precision, recall, F1 and a confusion matrix on a held-out test split — so quality is **measured, not claimed**.
- **A test suite + CI** (pytest + GitHub Actions) that runs on every push.
- **Data augmentation**, structured **logging**, a **config module**, and a **pre-flight environment checker**.
- An analytics dashboard, Excel export, and a student registry on top of the live app.

The classical engine is the proven default and needs zero downloads; the deep engine activates after a one-time 37 MB model fetch and degrades gracefully if absent.

---

## The Two Engines

| | Classical | Deep Learning |
|---|---|---|
| **Detector** | Haar Cascade (2001) | **YuNet** CNN (2023) |
| **Recognizer** | LBPH histograms (2006) | **SFace** CNN embeddings (2021) |
| **Decision rule** | Chi-square *distance* | 128-D embedding **cosine similarity** |
| **Training** | Instant, per-class texture model | Builds a per-student embedding gallery |
| **Strengths** | Tiny, instant, no weights | Robust to pose/lighting, scales to many identities |
| **Hardware** | CPU | CPU (no GPU needed) |
| **Paradigm** | Hand-crafted features | Learned embeddings — same family as FaceNet / ArcFace |

Both run through the same `RecognitionPipeline`, which normalises their different score scales (distance vs. similarity) into one confidence value and one GUI threshold slider.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                            main.py                            │
│                  Tkinter root window + lifecycle              │
└───────────────────────────────┬───────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │       app.py        │   GUI controller — never
                     │   AttendanceApp     │   touches OpenCV directly
                     └──────────┬──────────┘
                                │ uses
                     ┌──────────▼───────────────┐
                     │  core/pipeline.py         │
                     │  RecognitionPipeline      │  ← engine registry,
                     │  (detector + recognizer)  │     graceful fallback,
                     └───┬──────────────────┬────┘     unified threshold
                         │                  │
          ┌──────────────▼───┐     ┌────────▼──────────────┐
          │ core/detectors.py│     │ core/recognizers.py    │
          │  HaarDetector    │     │  LBPHRecognizer        │
          │  YuNetDetector   │     │  SFaceRecognizer       │
          └──────────────────┘     └────────────────────────┘
                         │                  │
                  ┌──────▼──────┐    ┌───────▼─────────┐
                  │core/dataset │    │core/augmentation│
                  └─────────────┘    └─────────────────┘

  Built on the same core:
     evaluate.py  → metrics + confusion matrix    tests/ → pytest + CI
     analytics.py → matplotlib dashboard          config.py → all constants
```

**Per-frame recognition pipeline:**

```
Camera frame ─► Detector (Haar | YuNet) ─► face boxes
                                              │
                                   Recognizer (LBPH | SFace)
                                              │
                          ┌───────────────────┴───────────────────┐
                   recognized (≥ threshold)              below threshold
                          │                                       │
              mark in CSV + audio beep                    show "Unknown"
```

---

## Live Results — Measured, Not Claimed

Generate these yourself on your own enrolled dataset:

```bash
python evaluate.py                 # classical engine (Haar + LBPH)
python evaluate.py --engine deep   # deep engine (YuNet + SFace)
python evaluate.py --augment       # with training-time augmentation
```

Each run performs a stratified 70/30 train/test split per student and prints a full `scikit-learn` classification report plus an overall accuracy, and saves a confusion matrix to `assets/confusion_matrix_<engine>.png`. The interactive walkthrough lives in [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb).

> Methodology note: during the Month 1 feasibility study, LBPH reached ~94% on a 10-student set and was chosen over Eigenfaces (88%) and FisherFaces (91%) for its robustness to lighting. The deep engine was added later to push accuracy and scalability further; run the evaluator to reproduce the comparison on your data.

---

## Features

| Area | Feature |
|---|---|
| Recognition | Two engines (classical + deep), switchable live from the toolbar |
| Recognition | Unified confidence threshold slider, auto-ranged per engine |
| Detection | Haar Cascade *or* YuNet CNN with landmark-based alignment |
| Enrollment | One-click capture of 50 images with diversity throttling |
| Robustness | Optional training-time **data augmentation** (flip, brightness, rotation) |
| Attendance | Real-time marking, audio beep, live "marked today" counter, duplicate-safe |
| Records | Dated CSV, color-coded table, live name search, per-session & bulk Excel export |
| Analytics | matplotlib dashboard: daily trend, per-student %, summary metrics |
| Admin | Student registry — view image counts, delete a student with full cleanup |
| Engineering | `evaluate.py` metrics, pytest suite, GitHub Actions CI, logging, config module |
| Onboarding | `check_env.py` pre-flight check, `download_models.py` weight fetcher |

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Classical detection / recognition | OpenCV Haar Cascade · LBPH |
| Deep detection / recognition | OpenCV `FaceDetectorYN` (YuNet) · `FaceRecognizerSF` (SFace) |
| GUI | Tkinter + Pillow |
| Data / records | pandas · openpyxl |
| Analytics & evaluation | matplotlib · scikit-learn |
| Testing & CI | pytest · ruff · GitHub Actions |
| Camera backend | DirectShow (Windows) |

---

## Project Structure

```
Exploratory/
│
├── main.py                  # Entry point
├── app.py                   # GUI controller (no direct OpenCV calls)
├── config.py                # Every tuneable constant
│
├── core/                    # Framework-agnostic recognition engine
│   ├── detectors.py         #   Haar + YuNet detectors
│   ├── recognizers.py       #   LBPH + SFace recognizers
│   ├── pipeline.py          #   engine orchestration + fallback
│   ├── augmentation.py      #   training-time augmentation
│   └── dataset.py           #   dataset loading / label parsing
│
├── attendance_manager.py    # Daily CSV logic + Excel export
├── analytics.py             # matplotlib analytics window
├── student_registry.py      # Student registry window
├── face_trainer.py          # Back-compat shim → core
│
├── evaluate.py              # Reproducible metrics + confusion matrix
├── download_models.py       # Fetch YuNet + SFace weights (OpenCV Zoo)
├── check_env.py             # Pre-flight dependency + webcam check
│
├── tests/                   # pytest suite (augmentation, manager, recognizers, pipeline)
├── notebooks/
│   └── model_evaluation.ipynb
├── .github/workflows/ci.yml # Lint + test on every push
├── pyproject.toml           # ruff + pytest config
├── requirements.txt
├── requirements-dev.txt
│
├── dataset/    trainer/    Attendance/    models/    (all git-ignored, runtime)
└── LICENSE   README.md
```

---

## Setup & Installation

```bash
# 1 — clone
git clone https://github.com/yashtahlyani/Exploratory.git
cd Exploratory

# 2 — install
pip install -r requirements.txt

# 3 — verify environment (libraries + webcam)
python check_env.py

# 4 — (optional) enable the deep-learning engine — one-time ~37 MB download
python download_models.py

# 5 — run
python main.py
```

> **Prerequisites:** Python 3.11+, a webcam, and Windows (DirectShow backend). The classical engine works immediately; step 4 is only needed for the deep engine.

---

## How to Use

1. **Register** — in the Registration panel enter ID + name, **Start Camera**, **Take Images** (captures 50), then **Train & Save**. Tick *Augment data* for extra robustness.
2. **Pick an engine** — choose *Classical* or *Deep Learning* from the toolbar dropdown (deep appears once weights are downloaded). Tune the **threshold** slider if needed.
3. **Mark attendance** — **Take Attendance**; recognized students are marked automatically with an audio beep and a live counter.
4. **Review & export** — **View Records** to browse/search by name and export to Excel; **Export All → Excel** for one workbook across all dates.
5. **Analyse** — **Analytics** for trends and per-student rates; **Manage Students** to audit or remove enrollments.
6. **Benchmark** — `python evaluate.py [--engine deep] [--augment]` to measure model quality.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest          # run the test suite
ruff check .    # lint
```

CI runs both on every push and pull request via GitHub Actions.

---

## Configuration

Everything tuneable lives in `config.py` — engine defaults, thresholds and ranges per engine, capture settings, paths, model URLs, and the UI palette. No logic changes needed to retune the system.

---

## Limitations & Future Work

| Limitation | Direction |
|---|---|
| Windows-only camera backend | Abstract DirectShow vs. V4L2 for Linux / macOS |
| No liveness detection | Add anti-spoofing to reject photo/video replay attacks |
| Manual retrain per engine | Background auto-retrain after each enrollment |
| Embeddings averaged into one gallery vector | Store k-NN over all embeddings for harder cases |
| Single camera | Multi-camera support for large lecture halls |
| Local-only records | Web dashboard for faculty cross-date analytics |

---

## Team

| Name | Roll Number |
|---|---|
| Hardik Dochania | 24045055 |
| Yagya Gupta | 24045146 |
| Yash Tahlyani | 24045147 |

**Supervisors:** Prof. Pradeep Kumar · Prof. Durga Prasad  
**Teaching Assistant:** Mr. Kushagra Singh

---

## License

Licensed under the **MIT License** — see [LICENSE](LICENSE).

---

<div align="center">

Made with dedication at **IIT (BHU) Varanasi**

</div>
