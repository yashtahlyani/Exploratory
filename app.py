import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime

from face_trainer import FaceTrainer
from attendance_manager import AttendanceManager

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DATASET_DIR = "dataset"
STUDENT_CSV = "student_details.csv"

# --- tuning constants ---
CONFIDENCE_THRESHOLD = 80   # LBPH: lower=better match; <80 = recognized
MIN_FACE_PX = 50            # ignore faces smaller than this (px)
CAPTURE_EVERY_N = 4         # save 1 image every N frames for diversity

# --- colours ---
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#21262d"
ACCENT    = "#e94560"
ACCENT2   = "#1f6feb"
FG_WHITE  = "#f0f6fc"
FG_MUTED  = "#8b949e"
GREEN     = "#3fb950"
ORANGE    = "#d29922"


class AttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition Based Attendance System")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 640)
        self.root.configure(bg=BG_DARK)

        self.trainer      = FaceTrainer()
        self.att_mgr      = AttendanceManager()
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        # cameras
        self.att_cam  = None
        self.reg_cam  = None

        # state
        self.att_running     = False
        self.reg_cam_on      = False
        self.is_capturing    = False
        self.img_count       = 0
        self._cap_frame_tick = 0          # frame counter for diversity throttle
        self.recognizer      = None
        self._student_map: dict = {}      # cached — refreshed on attendance start

        self.student_id   = tk.StringVar()
        self.student_name = tk.StringVar()

        self._build_ui()
        self._refresh_reg_count()
        self._tick_clock()

    # ================================================================ UI =====

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._build_attendance_panel(body)
        self._build_registration_panel(body)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=66)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=BG_PANEL)
        left.pack(side="left", padx=20, fill="y")
        tk.Label(left, text="Face Recognition Based Attendance System",
                 font=("Segoe UI", 15, "bold"), fg=FG_WHITE, bg=BG_PANEL
                 ).pack(anchor="w", pady=(12, 0))
        tk.Label(left,
                 text="IIT (BHU) Varanasi  |  Hardik  •  Yagya Gupta  •  Yash Tahlyani",
                 font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_PANEL
                 ).pack(anchor="w")

        right = tk.Frame(hdr, bg=BG_PANEL)
        right.pack(side="right", padx=20, fill="y")
        self.clock_lbl = tk.Label(right, font=("Segoe UI", 13), fg=ACCENT, bg=BG_PANEL)
        self.clock_lbl.pack(anchor="e", pady=(12, 0))
        self.date_lbl  = tk.Label(right, font=("Segoe UI", 9),  fg=FG_MUTED, bg=BG_PANEL)
        self.date_lbl.pack(anchor="e")

    def _build_attendance_panel(self, parent):
        f = tk.Frame(parent, bg=BG_PANEL, highlightthickness=1,
                     highlightbackground="#30363d")
        f.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=16)

        tk.Frame(f, bg=ACCENT, height=4).pack(fill="x")
        tk.Label(f, text="ATTENDANCE", font=("Segoe UI", 12, "bold"),
                 fg=FG_WHITE, bg=BG_PANEL).pack(pady=(10, 2))
        tk.Label(f, text="For Already Registered Students",
                 font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_PANEL).pack()

        cam_wrap = tk.Frame(f, bg="black", highlightthickness=1,
                            highlightbackground="#30363d")
        cam_wrap.pack(padx=14, pady=10, fill="both", expand=True)
        self.att_cam_lbl = tk.Label(cam_wrap, bg="black",
                                    text="Camera Off", fg=FG_MUTED,
                                    font=("Segoe UI", 12))
        self.att_cam_lbl.pack(fill="both", expand=True)

        self.att_status = tk.Label(f, text="Press Take Attendance to begin",
                                   font=("Segoe UI", 10), fg=FG_MUTED, bg=BG_PANEL)
        self.att_status.pack(pady=4)

        btn_row = tk.Frame(f, bg=BG_PANEL)
        btn_row.pack(pady=(2, 14))
        self.att_btn = self._btn(btn_row, "Take Attendance", ACCENT,
                                 self._toggle_attendance)
        self.att_btn.pack(side="left", padx=6)
        self._btn(btn_row, "View Records", ACCENT2,
                  self._open_records_window).pack(side="left", padx=6)

    def _build_registration_panel(self, parent):
        f = tk.Frame(parent, bg=BG_PANEL, highlightthickness=1,
                     highlightbackground="#30363d")
        f.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=16)

        tk.Frame(f, bg=ACCENT2, height=4).pack(fill="x")
        tk.Label(f, text="REGISTRATION", font=("Segoe UI", 12, "bold"),
                 fg=FG_WHITE, bg=BG_PANEL).pack(pady=(10, 2))
        tk.Label(f, text="For New Student Enrollment",
                 font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_PANEL).pack()

        cam_wrap = tk.Frame(f, bg="black", highlightthickness=1,
                            highlightbackground="#30363d", height=240)
        cam_wrap.pack(padx=14, pady=8, fill="x")
        cam_wrap.pack_propagate(False)   # fixed height — never pushes form down
        self.reg_cam_lbl = tk.Label(cam_wrap, bg="black",
                                    text="Camera Off", fg=FG_MUTED,
                                    font=("Segoe UI", 12))
        self.reg_cam_lbl.pack(fill="both", expand=True)

        form = tk.Frame(f, bg=BG_PANEL)
        form.pack(padx=14, fill="x")
        self._form_row(form, "Enter ID:",   self.student_id,   0)
        self._form_row(form, "Enter Name:", self.student_name, 1)
        form.columnconfigure(1, weight=1)

        self.reg_capture_status = tk.Label(
            f, text="", font=("Segoe UI", 10, "bold"),
            fg=ORANGE, bg=BG_PANEL)
        self.reg_capture_status.pack(pady=(4, 0))

        self.img_count_lbl = tk.Label(f, text="Images Captured: 0 / 50",
                                      font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_PANEL)
        self.img_count_lbl.pack()
        self.reg_count_lbl = tk.Label(f, text="Total Registrations: 0",
                                      font=("Segoe UI", 9), fg=GREEN, bg=BG_PANEL)
        self.reg_count_lbl.pack(pady=(0, 6))

        btn_row = tk.Frame(f, bg=BG_PANEL)
        btn_row.pack(pady=(0, 14))

        self.cam_btn = self._btn(btn_row, "Start Camera", "#533483",
                                 self._toggle_reg_camera)
        self.cam_btn.pack(side="left", padx=4)

        self.cap_btn = self._btn(btn_row, "Take Images", ACCENT,
                                 self._start_capture)
        self.cap_btn.pack(side="left", padx=4)
        self.cap_btn.config(state="disabled")

        self.train_btn = self._btn(btn_row, "Train & Save", ACCENT2,
                                   self._train_and_save)
        self.train_btn.pack(side="left", padx=4)

    # ============================================================= helpers ===

    @staticmethod
    def _btn(parent, text, color, cmd, **kw):
        return tk.Button(parent, text=text, font=("Segoe UI", 10, "bold"),
                         bg=color, fg="white", activebackground=color,
                         activeforeground="white", relief="flat",
                         padx=14, pady=7, cursor="hand2", command=cmd,
                         bd=0, **kw)

    @staticmethod
    def _form_row(parent, label, var, row):
        tk.Label(parent, text=label, font=("Segoe UI", 10), fg=FG_WHITE,
                 bg=BG_PANEL, width=12, anchor="w"
                 ).grid(row=row, column=0, padx=(0, 8), pady=5, sticky="w")
        tk.Entry(parent, textvariable=var, font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_WHITE, insertbackground=FG_WHITE,
                 relief="flat", bd=6
                 ).grid(row=row, column=1, pady=5, sticky="ew")

    def _tick_clock(self):
        now = datetime.now()
        self.clock_lbl.config(text=now.strftime("%H:%M:%S"))
        self.date_lbl.config(text=now.strftime("%A, %d %B %Y"))
        self.root.after(1000, self._tick_clock)

    def _refresh_reg_count(self):
        count = 0
        if os.path.exists(DATASET_DIR):
            ids = {f.split(".")[1] for f in os.listdir(DATASET_DIR)
                   if f.lower().endswith(".jpg") and len(f.split(".")) >= 3}
            count = len(ids)
        self.reg_count_lbl.config(text=f"Total Registrations: {count}")

    def _frame_to_imgtk(self, frame, label_widget, fixed_h=None):
        w = label_widget.winfo_width()  or 480
        h = fixed_h or label_widget.winfo_height() or 300
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((max(w, 80), max(h, 60)), Image.LANCZOS)
        return ImageTk.PhotoImage(image=img)

    def _open_camera(self):
        """Open camera with DirectShow (reliable on Windows)."""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Could not open webcam.\n"
                                 "Make sure it is connected and not in use.")
            return None
        # test a read
        ret, _ = cap.read()
        if not ret:
            cap.release()
            messagebox.showerror("Camera Error",
                                 "Camera opened but returned no frames.\n"
                                 "Try closing other apps using the webcam.")
            return None
        return cap

    def _detect_faces(self, gray):
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            minSize=(MIN_FACE_PX, MIN_FACE_PX)
        )

    def _save_student(self, sid: str, name: str):
        try:
            sid_int = int(sid)
        except ValueError:
            return
        if os.path.exists(STUDENT_CSV):
            df = pd.read_csv(STUDENT_CSV)
            if sid_int in df["ID"].values:
                return
        else:
            df = pd.DataFrame(columns=["ID", "Name"])
        df = pd.concat([df, pd.DataFrame({"ID": [sid_int], "Name": [name]})],
                       ignore_index=True)
        df.to_csv(STUDENT_CSV, index=False)

    def _load_student_map(self) -> dict:
        if os.path.exists(STUDENT_CSV):
            df = pd.read_csv(STUDENT_CSV)
            return dict(zip(df["ID"].astype(int), df["Name"]))
        return {}

    # ======================================================== ATTENDANCE ====

    def _toggle_attendance(self):
        if self.att_running:
            self._stop_attendance()
        else:
            self._start_attendance()

    def _start_attendance(self):
        if not self.trainer.model_exists():
            messagebox.showerror("No Model",
                                 "No trained model found.\n"
                                 "Register students and click Train & Save first.")
            return

        # Stop registration camera if running — avoid dual-camera conflict
        if self.reg_cam_on:
            self._stop_reg_camera()

        cam = self._open_camera()
        if cam is None:
            return

        self.att_cam = cam
        self.recognizer = self.trainer.load_recognizer()
        self._student_map = self._load_student_map()   # cache once per session
        self.att_running = True
        self.att_btn.config(text="Stop", bg=GREEN)
        self.att_status.config(text="Scanning…", fg=FG_MUTED)
        self._att_loop()

    def _stop_attendance(self):
        self.att_running = False
        if self.att_cam:
            self.att_cam.release()
            self.att_cam = None
        self.att_btn.config(text="Take Attendance", bg=ACCENT)
        self.att_cam_lbl.config(image="", text="Camera Off", fg=FG_MUTED)
        self.att_status.config(text="Session ended.", fg=FG_MUTED)

    def _att_loop(self):
        if not self.att_running or self.att_cam is None:
            return

        ret, frame = self.att_cam.read()
        if not ret:
            self.root.after(30, self._att_loop)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        status_text  = "No face detected"
        status_color = FG_MUTED

        for (x, y, w, h) in faces:
            face_roi = FaceTrainer.prepare_face(gray[y: y + h, x: x + w])
            sid, confidence = self.recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name        = self._student_map.get(sid, "Unknown")
                conf_pct    = f"{round(100 - confidence)}%"
                color       = (50, 205, 50)
                newly       = self.att_mgr.mark_attendance(sid, name)
                status_text  = (f"Marked: {name} ({conf_pct})"
                                if newly else f"Already marked: {name} ({conf_pct})")
                status_color = GREEN
            else:
                name         = "Unknown"
                conf_pct     = f"conf:{round(confidence)}"
                color        = (60, 60, 220)
                status_text  = "Face detected — not recognized"
                status_color = ORANGE

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name,     (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, conf_pct, (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,  color, 2)

        self.att_status.config(text=status_text, fg=status_color)
        imgtk = self._frame_to_imgtk(frame, self.att_cam_lbl)
        self.att_cam_lbl.imgtk = imgtk
        self.att_cam_lbl.config(image=imgtk, text="")

        self.root.after(30, self._att_loop)

    # ===================================================== REGISTRATION =====

    def _toggle_reg_camera(self):
        if self.reg_cam_on:
            self._stop_reg_camera()
        else:
            self._start_reg_camera()

    def _start_reg_camera(self):
        # Stop attendance camera if running
        if self.att_running:
            self._stop_attendance()

        cam = self._open_camera()
        if cam is None:
            return

        self.reg_cam = cam
        self.reg_cam_on = True
        self.cam_btn.config(text="Stop Camera", bg=GREEN)
        self.cap_btn.config(state="normal")
        self._reg_preview_loop()

    def _stop_reg_camera(self):
        self.reg_cam_on  = False
        self.is_capturing = False
        if self.reg_cam:
            self.reg_cam.release()
            self.reg_cam = None
        self.cam_btn.config(text="Start Camera", bg="#533483")
        self.cap_btn.config(state="disabled")
        self.reg_cam_lbl.config(image="", text="Camera Off", fg=FG_MUTED)
        self.reg_capture_status.config(text="")

    def _start_capture(self):
        sid  = self.student_id.get().strip()
        name = self.student_name.get().strip()
        if not sid:
            messagebox.showerror("Missing Field", "Please enter a Student ID.")
            return
        if not name:
            messagebox.showerror("Missing Field", "Please enter a Student Name.")
            return
        try:
            int(sid)
        except ValueError:
            messagebox.showerror("Invalid ID", "Student ID must be a number.")
            return
        if self.reg_cam is None:
            messagebox.showerror("Camera", "Start the camera first.")
            return

        os.makedirs(DATASET_DIR, exist_ok=True)
        self._save_student(sid, name)
        self.img_count       = 0
        self._cap_frame_tick = 0
        self.is_capturing    = True
        self.cap_btn.config(text="Capturing…", state="disabled", bg=ORANGE)
        self.reg_capture_status.config(text="Look at the camera…", fg=ORANGE)

    def _reg_preview_loop(self):
        if not self.reg_cam_on or self.reg_cam is None:
            return

        ret, frame = self.reg_cam.read()
        if not ret:
            self.root.after(30, self._reg_preview_loop)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        if self.is_capturing:
            self._cap_frame_tick += 1

            if len(faces) == 0:
                self.reg_capture_status.config(
                    text="No face detected — move closer", fg=ORANGE)
            else:
                self.reg_capture_status.config(
                    text=f"Capturing… {self.img_count}/50", fg=GREEN)

            # save one face every CAPTURE_EVERY_N frames for image diversity
            if self._cap_frame_tick % CAPTURE_EVERY_N == 0 and len(faces) > 0:
                x, y, w, h = faces[0]   # use the largest/first face only
                self.img_count += 1
                face_gray = gray[y: y + h, x: x + w]
                path = os.path.join(
                    DATASET_DIR,
                    f"User.{self.student_id.get().strip()}.{self.img_count}.jpg"
                )
                cv2.imwrite(path, face_gray)

            self.img_count_lbl.config(
                text=f"Images Captured: {self.img_count} / 50")

            if self.img_count >= 50:
                self.is_capturing = False
                self.cap_btn.config(text="Take Images", state="normal", bg=ACCENT)
                self.reg_capture_status.config(text="50 images captured!", fg=GREEN)
                self._refresh_reg_count()
                messagebox.showinfo(
                    "Capture Complete",
                    f"50 images captured for {self.student_name.get().strip()}!\n\n"
                    "Click  Train & Save  to update the recognition model."
                )

        # draw rectangles on preview
        for (x, y, w, h) in faces:
            color = (0, 200, 0) if self.is_capturing else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if self.is_capturing:
                cv2.putText(frame, f"{self.img_count}/50",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

        imgtk = self._frame_to_imgtk(frame, self.reg_cam_lbl, fixed_h=240)
        self.reg_cam_lbl.imgtk = imgtk
        self.reg_cam_lbl.config(image=imgtk, text="")

        self.root.after(30, self._reg_preview_loop)

    def _train_and_save(self):
        if not os.path.exists(DATASET_DIR) or not any(
            f.lower().endswith(".jpg") for f in os.listdir(DATASET_DIR)
        ):
            messagebox.showerror(
                "No Dataset",
                "Dataset folder is empty.\nCapture images for at least one student first."
            )
            return

        # disable buttons during training
        self.train_btn.config(text="Training…", state="disabled", bg=FG_MUTED)
        self.root.update()

        try:
            count = self.trainer.train()
            self.train_btn.config(text="Train & Save", state="normal", bg=ACCENT2)
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully on {count} student(s)!\n"
                "You can now use Take Attendance."
            )
            self._refresh_reg_count()
        except Exception as exc:
            self.train_btn.config(text="Train & Save", state="normal", bg=ACCENT2)
            messagebox.showerror("Training Failed", str(exc))

    # ======================================================== RECORDS ========

    def _open_records_window(self):
        files = self.att_mgr.list_attendance_files()
        if not files:
            messagebox.showinfo("No Records", "No attendance records found yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Attendance Records")
        win.geometry("780x520")
        win.configure(bg=BG_DARK)
        win.grab_set()

        tk.Label(win, text="Attendance Records",
                 font=("Segoe UI", 15, "bold"), fg=FG_WHITE, bg=BG_DARK
                 ).pack(pady=12)

        top = tk.Frame(win, bg=BG_DARK)
        top.pack(fill="x", padx=20, pady=(0, 8))
        tk.Label(top, text="Date:", fg=FG_WHITE, bg=BG_DARK,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))
        file_var = tk.StringVar(value=files[0])
        combo = ttk.Combobox(top, textvariable=file_var, values=files,
                             width=36, state="readonly")
        combo.pack(side="left")

        # treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Records.Treeview",
                        background=BG_CARD, foreground=FG_WHITE, rowheight=26,
                        fieldbackground=BG_CARD, font=("Segoe UI", 10))
        style.configure("Records.Treeview.Heading",
                        background=BG_PANEL, foreground=FG_WHITE,
                        font=("Segoe UI", 10, "bold"))
        style.map("Records.Treeview", background=[("selected", ACCENT2)])

        cols = ("ID", "Name", "Time", "Status")
        tree_frame = tk.Frame(win, bg=BG_DARK)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=6)

        # scrollbar MUST be packed before the treeview that expands
        sb = ttk.Scrollbar(tree_frame, orient="vertical")
        sb.pack(side="right", fill="y")

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                            style="Records.Treeview",
                            yscrollcommand=sb.set)
        sb.config(command=tree.yview)

        col_widths = {"ID": 80, "Name": 220, "Time": 130, "Status": 100}
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=col_widths[col], anchor="center")
        tree.pack(side="left", fill="both", expand=True)

        count_lbl = tk.Label(win, text="", fg=GREEN, bg=BG_DARK,
                             font=("Segoe UI", 10))
        count_lbl.pack(pady=(0, 10))

        def load_records(*_):
            tree.delete(*tree.get_children())
            df = self.att_mgr.load_file(file_var.get())
            for _, row in df.iterrows():
                tag = "present" if str(row.get("Status", "")).lower() == "present" \
                      else "absent"
                tree.insert("", "end", values=tuple(row), tags=(tag,))
            tree.tag_configure("present", foreground=GREEN)
            tree.tag_configure("absent",  foreground=ACCENT)
            count_lbl.config(text=f"Total present today: {len(df)}")

        combo.bind("<<ComboboxSelected>>", load_records)
        load_records()

    # =========================================================== lifecycle ===

    def on_close(self):
        self._stop_attendance()
        self._stop_reg_camera()
        self.root.destroy()
