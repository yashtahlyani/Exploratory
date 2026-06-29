<div align="center">

# Face Recognition Based Attendance System

**Automated, contactless attendance marking using real-time face recognition**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Institute](https://img.shields.io/badge/IIT_(BHU)-Varanasi-B22222?style=for-the-badge)](https://iitbhu.ac.in)

<br/>

*Exploratory Project · Department of Computer Science & Engineering*
*Indian Institute of Technology (BHU) Varanasi*

</div>

---

## The Problem

Traditional classroom attendance is slow, error-prone, and breaks down at scale. A roll-call for 60 students takes 5–8 minutes of class time, proxy attendance is trivially easy, and paper sheets require manual digitisation. This project eliminates all of that: a single webcam marks every student automatically the moment they walk into frame — no buttons, no paper, no manual work.

---

## Solution Overview

A two-panel desktop application built entirely on commodity hardware and open-source libraries:

- **Registration** — enroll a student in under 2 minutes by capturing 50 facial images automatically
- **Attendance** — real-time recognition at ~30 fps with a live-adjustable confidence threshold
- **Records** — browse historical attendance by date with search/filter, export to Excel in one click
- **Analytics** — matplotlib-powered charts: daily attendance trends and per-student attendance rates
- **Student Registry** — view all enrolled students, see image counts, delete records with full cleanup

No internet connection. No cloud API. No GPU required. Runs entirely on a standard laptop.

---

## Screenshot

![Face Recognition Based Attendance System](assets/screenshot.png)

> Left panel handles live attendance scanning. Right panel handles student registration. Global toolbar gives access to Analytics, Student Registry, and bulk Excel export.

---

## Features

| Feature | Detail |
|---|---|
| Real-time face detection | OpenCV Haar Cascade, ~30 fps on CPU |
| Face recognition | LBPH — lightweight, offline, no cloud dependency |
| One-click registration | Captures 50 images automatically with diversity throttling |
| Instant attendance marking | No button press once camera is running |
| Audio beep on mark | Distinct audio feedback when a student is successfully marked |
| Live threshold slider | Adjust recognition strictness without restarting the session |
| Duplicate-safe | Each student marked once per day regardless of time in frame |
| Dated CSV records | One file per day, color-coded present/absent rows |
| Search / filter records | Live search by student name within the records window |
| Excel export | Per-session or bulk (all dates → one workbook) `.xlsx` export |
| Analytics dashboard | Daily trend bar chart + per-student attendance % horizontal bar |
| Student Registry | View enrolled students, image counts, delete with full cleanup |
| Camera conflict prevention | Opening one panel's camera auto-stops the other |
| Pre-flight env check | `check_env.py` verifies all dependencies + webcam before first run |
| Dark professional UI | Tkinter — no external UI framework needed |

---

## Why LBPH?

Three algorithms were evaluated during the Month 1 feasibility study:

| Algorithm | Accuracy (10-student test set) | Inference time | GPU required? |
|---|---|---|---|
| **LBPH** | **94%** | ~2 ms | No |
| FisherFaces | 91% | ~1 ms | No |
| Eigenfaces | 88% | ~1 ms | No |

LBPH was chosen because it is the most robust to lighting variation and partial occlusion while requiring zero GPU and only ~50 images per person to train reliably.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       main.py                            │
│              Tkinter root window entry point             │
└────────────────────────┬─────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │       app.py        │
              │   AttendanceApp     │   ← GUI controller (attendance
              │                     │     + registration + records
              └──┬───────────┬──────┘     + toolbar actions)
                 │           │
    ┌────────────▼──┐   ┌────▼──────────────┐
    │ face_trainer  │   │attendance_manager  │
    │ FaceTrainer   │   │ AttendanceManager  │
    │               │   │                    │
    │ • train()     │   │ • mark()           │
    │ • predict()   │   │ • export_xlsx()    │
    └───────┬───────┘   │ • export_all()     │
            │           └────────┬───────────┘
    ┌───────▼───────┐   ┌────────▼───────────┐
    │   dataset/    │   │    Attendance/      │
    │ User.ID.N.jpg │   │  DD-MM-YYYY.csv    │
    └───────────────┘   │  Attendance_All.xlsx│
                        └────────────────────┘

    ┌──────────────────────────────┐
    │       analytics.py           │     matplotlib charts
    │   AnalyticsWindow            │  ←  (daily trend,
    │   (Daily / Student / Summary)│      per-student %)
    └──────────────────────────────┘

    ┌──────────────────────────────┐
    │     student_registry.py      │     view + delete
    │   StudentRegistryWindow      │  ←  enrolled students
    └──────────────────────────────┘

    config.py  ←  single file for all tuneable constants
```

**Recognition pipeline per frame:**

```
Camera frame
     │
     ▼
Haar Cascade detector ──► No face? ──► Skip frame
     │
     ▼ (face ROI ≥ 50×50 px)
Resize to 100×100 px
     │
     ▼
LBPH Recognizer.predict()
     │
     ├── confidence < threshold ──► Recognized ──► Mark in CSV + audio beep
     │
     └── confidence ≥ threshold ──► Unknown    ──► Show warning on screen
```

---

## Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| Face Detection | OpenCV Haar Cascade | 4.13 |
| Face Recognition | OpenCV LBPH Recognizer | 4.13 |
| GUI | Tkinter (built-in) + Pillow | Python 3.11 |
| Analytics Charts | matplotlib (TkAgg backend) | 3.7+ |
| Attendance Records | pandas + CSV | 2.x |
| Excel Export | openpyxl | 3.x |
| Image Processing | NumPy + Pillow | 2.x |
| Camera Backend | DirectShow (Windows) | — |
| Audio Feedback | winsound (built-in) | — |

---

## Project Structure

```
Exploratory/
│
├── main.py                  # Entry point — launches the Tkinter app
├── app.py                   # Full GUI: all panels, toolbar, records window
├── face_trainer.py          # LBPH model training from captured face images
├── attendance_manager.py    # Daily CSV logic + Excel export (single & bulk)
├── analytics.py             # matplotlib Analytics window (3 tabs)
├── student_registry.py      # Student Registry window (view + delete)
├── config.py                # All tuneable constants in one place
├── check_env.py             # Pre-flight dependency + webcam checker
│
├── requirements.txt         # pip dependencies
├── LICENSE                  # MIT License
├── README.md
│
├── dataset/                 # Face images captured during registration (git-ignored)
│   └── User.<id>.<n>.jpg
├── trainer/                 # Trained LBPH model file (git-ignored)
│   └── trainer.yml
└── Attendance/              # Daily attendance files (git-ignored)
    ├── Attendance_DD-MM-YYYY.csv
    ├── Attendance_DD-MM-YYYY.xlsx
    └── Attendance_All.xlsx  # bulk export
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 or higher
- A working webcam
- Windows OS (uses DirectShow camera backend)

### 1 — Clone the repository

```bash
git clone https://github.com/yashtahlyani/Exploratory.git
cd Exploratory
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Verify your environment (recommended first-time)

```bash
python check_env.py
```

This checks all library imports and your webcam before you launch the app.

### 4 — Run the application

```bash
python main.py
```

---

## How to Use

### Step 1 — Register a Student

1. Open the **REGISTRATION** panel (right side of the window)
2. Enter the student's **numeric ID** and **Full Name**
3. Click **Start Camera** — the webcam activates with a live preview
4. Click **Take Images** — the system automatically captures 50 face images (spaced for diversity)
5. Click **Train & Save** — trains the LBPH model on all registered students

> Repeat steps 2–5 for each additional student. The model is retrained each time to include everyone.

### Step 2 — Mark Attendance

1. Open the **ATTENDANCE** panel (left side)
2. (Optional) Adjust the **Recognition Threshold** slider in the toolbar — lower = stricter
3. Click **Take Attendance** — the camera begins scanning
4. When a registered face is detected with sufficient confidence, attendance is marked automatically with an audio beep
5. The **Marked today** counter updates live
6. Click **Stop** when the session is over

### Step 3 — View & Export Records

1. Click **View Records** on the Attendance panel
2. Select a date from the dropdown
3. Type in the **Search** box to filter by student name in real time
4. Browse the color-coded table — **green** = Present, **red** = Unknown
5. Click **Export to Excel** to save the selected date, or use **Export All → Excel** in the toolbar

### Step 4 — Analytics

1. Click **Analytics** in the toolbar
2. **Daily Trend** tab — bar chart of present counts across the last 14 sessions
3. **Per Student** tab — horizontal bar chart of attendance % per enrolled student (75% threshold line shown)
4. **Summary** tab — key metrics: total sessions, avg rate, best/worst day, perfect attender

### Step 5 — Manage Students

1. Click **Manage Students** in the toolbar
2. View all enrolled students with their image counts (green = ≥50 images, orange = fewer)
3. Select a student and click **Delete Selected** to remove their record and all face images

---

## Configuration

All tuneable constants live in `config.py` — no logic changes required anywhere else:

```python
# Recognition
CONFIDENCE_THRESHOLD = 80   # default slider position (lower = stricter)
MIN_FACE_PX          = 50   # ignore faces smaller than 50×50 pixels

# Registration
CAPTURE_EVERY_N      = 4    # save 1 image every 4 frames (~7–8 fps at 30fps camera)
TARGET_IMAGES        = 50   # total images captured per student

# Face dimensions
FACE_SIZE = (100, 100)      # resize all faces to this before training and prediction
```

---

## Monthly Progress

| Month | Focus | Deliverable |
|---|---|---|
| **1** | Feasibility study, algorithm comparison (LBPH vs Eigenfaces vs FisherFaces), reference visit to IIT Ropar | Algorithm selection + student dataset (10 friends, ID card images) |
| **2** | Initial prototype, LBPH integration, dataset expansion, accuracy testing under varied lighting | Working recognition model, 50 images/student, ~94% accuracy |
| **3** | GUI polish, Analytics dashboard, Student Registry, Excel export, threshold slider, bug fixes | Final application — full-featured, stable, ready for demo |

---

## Limitations & Future Work

| Limitation | Potential Fix |
|---|---|
| Windows-only (DirectShow) | Replace `CAP_DSHOW` with `CAP_V4L2` for Linux / macOS |
| Single camera only | Multi-camera support for large lecture halls |
| No liveness detection | Anti-spoofing model to reject photo/video attacks |
| LBPH degrades with large student sets (>200) | Switch to deep learning embeddings (FaceNet / ArcFace) |
| No admin dashboard | Web-based dashboard for faculty to view cross-date analytics |
| Manual training step | Auto-retrain in background after every new registration |

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

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with dedication at **IIT (BHU) Varanasi**

</div>
