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

Traditional classroom attendance is slow, error-prone, and breaks down at scale. A roll-call for 60 students takes 5–8 minutes, proxy attendance is trivially easy, and paper sheets require manual digitisation. This project eliminates all of that: a single webcam marks every student automatically the moment they walk into frame.

---

## Solution Overview

A two-panel desktop application built entirely on commodity hardware and open-source libraries:

- **Registration panel** — enroll a student in under 2 minutes by capturing 50 facial images automatically
- **Attendance panel** — real-time recognition at ~30 fps; no button press required once the session starts
- **Records panel** — browse historical attendance by date, export to Excel in one click

No internet connection, no cloud API, no GPU required. Runs entirely on a standard laptop.

---

## Screenshot

![Face Recognition Based Attendance System](assets/screenshot.png)

> Left panel handles live attendance scanning. Right panel handles student registration — enter ID, name, capture face images, then train the model. Live clock and team credits always visible in the header.

---

## Key Features

| Feature | Detail |
|---|---|
| Real-time face detection | OpenCV Haar Cascade, ~30 fps on CPU |
| Face recognition | LBPH — lightweight, offline, no cloud dependency |
| One-click registration | Captures 50 images automatically with diversity throttling |
| Instant attendance marking | No button press once camera is running |
| Duplicate-safe | Each student marked once per day regardless of time in frame |
| Dated CSV records | One file per day, color-coded present/absent rows |
| Excel export | One-click `.xlsx` export with auto-sized columns |
| Camera conflict prevention | Opening one panel's camera auto-stops the other |
| Dark professional UI | Tkinter — no external UI framework needed |

---

## Why LBPH?

Three algorithms were evaluated during Month 1 feasibility study:

| Algorithm | Accuracy (10-student set) | Inference time | Requires GPU? |
|---|---|---|---|
| **LBPH** | 94% | ~2 ms | No |
| Eigenfaces | 88% | ~1 ms | No |
| FisherFaces | 91% | ~1 ms | No |

LBPH was chosen because it is the most robust to lighting variation and partial occlusion while requiring zero GPU and only ~50 images per person to train reliably.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│           Tkinter root window entry point           │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │       app.py        │
          │   AttendanceApp     │
          │   (GUI controller)  │
          └───┬─────────────┬───┘
              │             │
   ┌──────────▼──┐    ┌─────▼───────────┐
   │face_trainer │    │attendance_manager│
   │ FaceTrainer │    │AttendanceManager │
   │             │    │                  │
   │ • train()   │    │ • mark()         │
   │ • predict() │    │ • export_xlsx()  │
   └──────┬──────┘    └──────┬───────────┘
          │                  │
   ┌──────▼──────┐    ┌──────▼───────────┐
   │  dataset/   │    │   Attendance/    │
   │ User.ID.N   │    │  DD-MM-YYYY.csv  │
   │    .jpg     │    │  DD-MM-YYYY.xlsx │
   └─────────────┘    └──────────────────┘

Config: config.py — single place to tune thresholds, paths, and UI colours
```

**Recognition pipeline:**

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
     ├── confidence < 80 ──► Recognized ──► Mark attendance in CSV
     │
     └── confidence ≥ 80 ──► Unknown    ──► Show warning on screen
```

The LBPH (Local Binary Pattern Histogram) algorithm encodes texture patterns around each pixel into a histogram. It runs entirely on CPU, is invariant to monotonic grey-level changes (i.e. lighting shifts), and achieves reliable recognition with as few as 50 images per person.

---

## Tech Stack

| Component | Library / Tool | Version |
|---|---|---|
| Face Detection | OpenCV Haar Cascade | 4.13 |
| Face Recognition | OpenCV LBPH Recognizer | 4.13 |
| GUI | Tkinter (built-in) + Pillow | Python 3.11 |
| Attendance Records | pandas + CSV | 2.x |
| Excel Export | openpyxl | 3.x |
| Image Processing | NumPy + Pillow | 2.x |
| Camera Backend | DirectShow (Windows) | — |

---

## Project Structure

```
Exploratory/
│
├── main.py                  # Entry point — launches the Tkinter app
├── app.py                   # Full GUI: Attendance + Registration + Records panels
├── face_trainer.py          # LBPH model training from captured face images
├── attendance_manager.py    # Daily CSV logic (mark, load, list, export to Excel)
├── config.py                # All tuneable constants in one place
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
    └── Attendance_DD-MM-YYYY.xlsx
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

### 3 — Run the application

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
2. Click **Take Attendance** — the camera begins scanning
3. When a registered face is detected with sufficient confidence, attendance is marked automatically
4. The student's name, timestamp, and confidence score appear on screen
5. Click **Stop** when the session is over

### Step 3 — View & Export Records

1. Click **View Records** on the Attendance panel
2. Select a date from the dropdown
3. Browse the color-coded table — **green** = Present, **red** = Unknown
4. Click **Export to Excel** to save the selected date's records as a formatted `.xlsx` file

---

## Configuration

All tuneable constants live in `config.py` — no logic changes required:

```python
# Recognition
CONFIDENCE_THRESHOLD = 80   # lower = stricter match (LBPH: lower confidence = better)
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
| **3** | GUI polish, export feature, lighting robustness, bug fixes, deployment readiness | Final application — stable, user-friendly, ready for demo |

---

## Limitations & Future Work

| Limitation | Potential Fix |
|---|---|
| Windows-only (DirectShow) | Replace `CAP_DSHOW` with `CAP_V4L2` for Linux support |
| Single camera only | Multi-camera support for large lecture halls |
| No liveness detection | Anti-spoofing model (photo/video attacks) |
| LBPH degrades with large student sets (>200) | Switch to deep learning embeddings (FaceNet / ArcFace) |
| No admin dashboard | Web-based dashboard for faculty to view cross-date analytics |

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
