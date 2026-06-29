# Face Recognition Based Attendance System

An automated attendance system using real-time face recognition, built as an Exploratory Project at **IIT (BHU) Varanasi**.

**Team:** Hardik Dochania (24045055) · Yagya Gupta (24045146) · Yash Tahlyani (24045147)  
**Supervisors:** Prof. Pradeep Kumar · Prof. Durga Prasad

---

## Features

- Real-time face detection and recognition via webcam
- One-click student registration with automatic image capture (50 images/student)
- LBPH (Local Binary Pattern Histogram) face recognition model
- Attendance auto-saved to dated CSV files
- View historical attendance records by date inside the app
- Dark-themed, responsive Tkinter GUI

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python main.py
```

---

## How to Use

### Register a New Student
1. Open the **REGISTRATION** panel (right side).
2. Enter the student's **ID** (numeric) and **Name**.
3. Click **Start Camera** — the webcam will open.
4. Click **Take Images** — the system captures 50 face images automatically.
5. Click **Train & Save** — trains the recognition model on all registered students.

### Mark Attendance
1. Open the **ATTENDANCE** panel (left side).
2. Click **Take Attendance** — the camera starts scanning in real time.
3. When a registered face is detected with ≥45% confidence, attendance is marked.
4. Click **Stop** when done.

### View Records
- Click **View Records** to browse attendance by date in a table view.
- CSV files are stored in the `Attendance/` folder.

---

## Project Structure

```
├── main.py                  # Entry point
├── app.py                   # Tkinter GUI (both panels)
├── face_trainer.py          # LBPH model training
├── attendance_manager.py    # CSV attendance logic
├── requirements.txt
├── dataset/                 # Face images (auto-created, git-ignored)
├── trainer/                 # Trained model (auto-created, git-ignored)
└── Attendance/              # Daily CSV records (auto-created, git-ignored)
```

---

## Technology Stack

| Component | Library |
|-----------|---------|
| Face detection | OpenCV Haar Cascade |
| Face recognition | OpenCV LBPH Recognizer |
| GUI | Tkinter + Pillow |
| Attendance records | pandas + CSV |
| Image processing | NumPy + Pillow |

---

## Monthly Progress

| Month | Focus |
|-------|-------|
| 1 | Dataset collection, feasibility study, Excel structure |
| 2 | Initial prototype, LBPH integration, dataset expansion |
| 3 | GUI polishing, lighting tests, bug fixes, final deployment |
