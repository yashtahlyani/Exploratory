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

BG_DARK = "#0d1117"
BG_PANEL = "#161b22"
BG_CARD = "#21262d"
ACCENT = "#e94560"
ACCENT2 = "#1f6feb"
ACCENT3 = "#8b949e"
FG_WHITE = "#f0f6fc"
FG_MUTED = "#8b949e"
GREEN = "#3fb950"
ORANGE = "#d29922"


class AttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition Based Attendance System")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BG_DARK)

        self.trainer = FaceTrainer()
        self.attendance_mgr = AttendanceManager()
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        self.camera = None
        self.att_camera = None
        self.is_attendance_running = False
        self.is_reg_camera_on = False
        self.is_capturing = False
        self.img_count = 0
        self.recognizer = None

        self.student_id = tk.StringVar()
        self.student_name = tk.StringVar()

        self._build_ui()
        self._refresh_reg_count()
        self._tick_clock()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        self._build_header()
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._build_attendance_panel(content)
        self._build_registration_panel(content)

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=BG_PANEL)
        left.pack(side="left", padx=20, fill="y")

        tk.Label(
            left,
            text="Face Recognition Based Attendance System",
            font=("Segoe UI", 16, "bold"),
            fg=FG_WHITE,
            bg=BG_PANEL,
        ).pack(anchor="w", pady=(12, 0))

        tk.Label(
            left,
            text="IIT (BHU) Varanasi  |  Hardik  •  Yagya Gupta  •  Yash Tahlyani",
            font=("Segoe UI", 9),
            fg=FG_MUTED,
            bg=BG_PANEL,
        ).pack(anchor="w")

        right = tk.Frame(header, bg=BG_PANEL)
        right.pack(side="right", padx=20, fill="y")

        self.clock_label = tk.Label(
            right,
            font=("Segoe UI", 13),
            fg=ACCENT,
            bg=BG_PANEL,
        )
        self.clock_label.pack(anchor="e", pady=(14, 0))

        self.date_label = tk.Label(
            right,
            font=("Segoe UI", 9),
            fg=FG_MUTED,
            bg=BG_PANEL,
        )
        self.date_label.pack(anchor="e")

    def _build_attendance_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0, highlightthickness=1,
                         highlightbackground="#30363d")
        frame.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=16)

        # Section title
        title_bar = tk.Frame(frame, bg=ACCENT, height=4)
        title_bar.pack(fill="x")

        tk.Label(
            frame,
            text="ATTENDANCE",
            font=("Segoe UI", 12, "bold"),
            fg=FG_WHITE,
            bg=BG_PANEL,
        ).pack(pady=(12, 4))

        tk.Label(
            frame,
            text="For Already Registered Students",
            font=("Segoe UI", 9),
            fg=FG_MUTED,
            bg=BG_PANEL,
        ).pack()

        # Camera feed
        cam_frame = tk.Frame(frame, bg="black", bd=1, relief="flat",
                             highlightbackground="#30363d", highlightthickness=1)
        cam_frame.pack(padx=14, pady=10, fill="both", expand=True)

        self.att_cam_label = tk.Label(cam_frame, bg="black", text="Camera Off",
                                      fg=FG_MUTED, font=("Segoe UI", 12))
        self.att_cam_label.pack(fill="both", expand=True)

        # Status strip
        self.att_status = tk.Label(
            frame,
            text="Ready — press Take Attendance to begin",
            font=("Segoe UI", 10),
            fg=FG_MUTED,
            bg=BG_PANEL,
        )
        self.att_status.pack(pady=4)

        # Buttons
        btn_frame = tk.Frame(frame, bg=BG_PANEL)
        btn_frame.pack(pady=(4, 14))

        self.att_btn = self._btn(btn_frame, "Take Attendance", ACCENT,
                                 self._toggle_attendance)
        self.att_btn.pack(side="left", padx=6)

        self._btn(btn_frame, "View Records", ACCENT2,
                  self._open_records_window).pack(side="left", padx=6)

    def _build_registration_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0, highlightthickness=1,
                         highlightbackground="#30363d")
        frame.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=16)

        title_bar = tk.Frame(frame, bg=ACCENT2, height=4)
        title_bar.pack(fill="x")

        tk.Label(
            frame,
            text="REGISTRATION",
            font=("Segoe UI", 12, "bold"),
            fg=FG_WHITE,
            bg=BG_PANEL,
        ).pack(pady=(12, 4))

        tk.Label(
            frame,
            text="For New Student Enrollment",
            font=("Segoe UI", 9),
            fg=FG_MUTED,
            bg=BG_PANEL,
        ).pack()

        # Camera feed
        cam_frame = tk.Frame(frame, bg="black", bd=1, relief="flat",
                             highlightbackground="#30363d", highlightthickness=1)
        cam_frame.pack(padx=14, pady=10, fill="both", expand=True)

        self.reg_cam_label = tk.Label(cam_frame, bg="black", text="Camera Off",
                                      fg=FG_MUTED, font=("Segoe UI", 12))
        self.reg_cam_label.pack(fill="both", expand=True)

        # Form
        form = tk.Frame(frame, bg=BG_PANEL)
        form.pack(padx=14, fill="x")

        self._form_row(form, "Enter ID:", self.student_id, 0)
        self._form_row(form, "Enter Name:", self.student_name, 1)
        form.columnconfigure(1, weight=1)

        # Progress label
        self.img_count_label = tk.Label(
            frame,
            text="Images Captured: 0 / 50",
            font=("Segoe UI", 9),
            fg=FG_MUTED,
            bg=BG_PANEL,
        )
        self.img_count_label.pack(pady=(6, 2))

        self.reg_count_label = tk.Label(
            frame,
            text="Total Registrations: 0",
            font=("Segoe UI", 9),
            fg=GREEN,
            bg=BG_PANEL,
        )
        self.reg_count_label.pack(pady=(0, 6))

        # Buttons
        btn_frame = tk.Frame(frame, bg=BG_PANEL)
        btn_frame.pack(pady=(0, 14))

        self.cam_toggle_btn = self._btn(btn_frame, "Start Camera", "#533483",
                                        self._toggle_reg_camera)
        self.cam_toggle_btn.pack(side="left", padx=4)

        self.capture_btn = self._btn(btn_frame, "Take Images", ACCENT,
                                     self._start_capture)
        self.capture_btn.pack(side="left", padx=4)
        self.capture_btn.config(state="disabled")

        self._btn(btn_frame, "Train & Save", ACCENT2,
                  self._train_and_save).pack(side="left", padx=4)

    # --------------------------------------------------------------- helpers -

    @staticmethod
    def _btn(parent, text, color, cmd):
        return tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            command=cmd,
            bd=0,
        )

    @staticmethod
    def _form_row(parent, label, var, row):
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10),
            fg=FG_WHITE,
            bg=BG_PANEL,
            width=12,
            anchor="w",
        ).grid(row=row, column=0, padx=(0, 8), pady=5, sticky="w")

        entry = tk.Entry(
            parent,
            textvariable=var,
            font=("Segoe UI", 10),
            bg=BG_CARD,
            fg=FG_WHITE,
            insertbackground=FG_WHITE,
            relief="flat",
            bd=6,
        )
        entry.grid(row=row, column=1, padx=0, pady=5, sticky="ew")
        return entry

    def _tick_clock(self):
        now = datetime.now()
        self.clock_label.config(text=now.strftime("%H:%M:%S"))
        self.date_label.config(text=now.strftime("%A, %d %B %Y"))
        self.root.after(1000, self._tick_clock)

    def _refresh_reg_count(self):
        count = 0
        if os.path.exists(DATASET_DIR):
            ids = {f.split(".")[1] for f in os.listdir(DATASET_DIR)
                   if f.endswith(".jpg") and len(f.split(".")) >= 3}
            count = len(ids)
        self.reg_count_label.config(text=f"Total Registrations: {count}")

    def _frame_to_imgtk(self, frame, size):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(image=img)

    def _load_student_map(self) -> dict:
        if os.path.exists(STUDENT_CSV):
            df = pd.read_csv(STUDENT_CSV)
            return dict(zip(df["ID"].astype(int), df["Name"]))
        return {}

    def _save_student(self, sid: str, name: str):
        csv_path = STUDENT_CSV
        try:
            sid_int = int(sid)
        except ValueError:
            return
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if sid_int in df["ID"].values:
                return
        else:
            df = pd.DataFrame(columns=["ID", "Name"])
        new_row = pd.DataFrame({"ID": [sid_int], "Name": [name]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)

    # ------------------------------------------------- attendance tab logic --

    def _toggle_attendance(self):
        if self.is_attendance_running:
            self._stop_attendance()
        else:
            self._start_attendance()

    def _start_attendance(self):
        if not self.trainer.model_exists():
            messagebox.showerror(
                "No Model",
                "No trained model found.\nPlease register students and click Train & Save first."
            )
            return

        self.recognizer = self.trainer.load_recognizer()
        self.att_camera = cv2.VideoCapture(0)
        if not self.att_camera.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return

        self.is_attendance_running = True
        self.att_btn.config(text="Stop", bg=GREEN)
        self.att_status.config(text="Scanning…", fg=FG_MUTED)
        self._att_loop()

    def _stop_attendance(self):
        self.is_attendance_running = False
        if self.att_camera:
            self.att_camera.release()
            self.att_camera = None
        self.att_btn.config(text="Take Attendance", bg=ACCENT)
        self.att_cam_label.config(image="", text="Camera Off", fg=FG_MUTED)
        self.att_status.config(text="Attendance session ended.", fg=FG_MUTED)

    def _att_loop(self):
        if not self.is_attendance_running or self.att_camera is None:
            return

        ret, frame = self.att_camera.read()
        if not ret:
            self.root.after(30, self._att_loop)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        student_map = self._load_student_map()

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            sid, confidence = self.recognizer.predict(face_roi)

            if confidence < 55:
                name = student_map.get(sid, "Unknown")
                newly_marked = self.attendance_mgr.mark_attendance(sid, name)
                label = f"{name}"
                conf_pct = f"{round(100 - confidence)}%"
                color = (50, 205, 50)
                status_text = (
                    f"Marked: {name} ({conf_pct})" if newly_marked
                    else f"Already marked: {name}"
                )
                self.att_status.config(text=status_text, fg=GREEN)
            else:
                label = "Unknown"
                conf_pct = f"{round(100 - confidence)}%"
                color = (60, 60, 220)
                self.att_status.config(text="Face detected but not recognized", fg=ORANGE)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, conf_pct, (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Resize to fit panel
        w_cam = self.att_cam_label.winfo_width() or 480
        h_cam = self.att_cam_label.winfo_height() or 300
        imgtk = self._frame_to_imgtk(frame, (max(w_cam, 80), max(h_cam, 60)))
        self.att_cam_label.imgtk = imgtk
        self.att_cam_label.config(image=imgtk, text="")

        self.root.after(30, self._att_loop)

    # ----------------------------------------------- registration tab logic --

    def _toggle_reg_camera(self):
        if self.is_reg_camera_on:
            self._stop_reg_camera()
        else:
            self._start_reg_camera()

    def _start_reg_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.is_reg_camera_on = True
        self.cam_toggle_btn.config(text="Stop Camera", bg=GREEN)
        self.capture_btn.config(state="normal")
        self._reg_preview_loop()

    def _stop_reg_camera(self):
        self.is_reg_camera_on = False
        self.is_capturing = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.cam_toggle_btn.config(text="Start Camera", bg="#533483")
        self.capture_btn.config(state="disabled")
        self.reg_cam_label.config(image="", text="Camera Off", fg=FG_MUTED)

    def _reg_preview_loop(self):
        if not self.is_reg_camera_on or self.camera is None:
            return

        ret, frame = self.camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)

            if self.is_capturing:
                for (x, y, w, h) in faces:
                    self.img_count += 1
                    face_gray = gray[y : y + h, x : x + w]
                    path = os.path.join(
                        DATASET_DIR,
                        f"User.{self.student_id.get()}.{self.img_count}.jpg"
                    )
                    cv2.imwrite(path, face_gray)
                    cv2.putText(
                        frame,
                        f"Capturing {self.img_count}/50",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

                self.img_count_label.config(
                    text=f"Images Captured: {self.img_count} / 50"
                )

                if self.img_count >= 50:
                    self.is_capturing = False
                    self.img_count = 0
                    messagebox.showinfo(
                        "Done",
                        f"50 images captured for {self.student_name.get()}!\n"
                        "Click Train & Save to update the model."
                    )
                    self._refresh_reg_count()

            w_lbl = self.reg_cam_label.winfo_width() or 480
            h_lbl = self.reg_cam_label.winfo_height() or 280
            imgtk = self._frame_to_imgtk(frame, (max(w_lbl, 80), max(h_lbl, 60)))
            self.reg_cam_label.imgtk = imgtk
            self.reg_cam_label.config(image=imgtk, text="")

        self.root.after(30, self._reg_preview_loop)

    def _start_capture(self):
        if not self.student_id.get().strip():
            messagebox.showerror("Missing Field", "Please enter a Student ID.")
            return
        if not self.student_name.get().strip():
            messagebox.showerror("Missing Field", "Please enter a Student Name.")
            return
        try:
            int(self.student_id.get().strip())
        except ValueError:
            messagebox.showerror("Invalid ID", "Student ID must be a number.")
            return
        if self.camera is None:
            messagebox.showerror("Camera", "Start the camera first.")
            return

        os.makedirs(DATASET_DIR, exist_ok=True)
        self._save_student(self.student_id.get().strip(), self.student_name.get().strip())
        self.img_count = 0
        self.is_capturing = True

    def _train_and_save(self):
        if not os.path.exists(DATASET_DIR) or not any(
            f.endswith(".jpg") for f in os.listdir(DATASET_DIR)
        ):
            messagebox.showerror(
                "No Dataset",
                "Dataset folder is empty.\nCapture images for at least one student first."
            )
            return

        try:
            count = self.trainer.train()
            messagebox.showinfo(
                "Training Complete",
                f"Model trained on {count} student(s) and saved successfully!"
            )
            self._refresh_reg_count()
        except Exception as exc:
            messagebox.showerror("Training Failed", str(exc))

    # --------------------------------------------------- records window ------

    def _open_records_window(self):
        files = self.attendance_mgr.list_attendance_files()
        if not files:
            messagebox.showinfo("No Records", "No attendance records found yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Attendance Records")
        win.geometry("760x520")
        win.configure(bg=BG_DARK)
        win.grab_set()

        tk.Label(
            win,
            text="Attendance Records",
            font=("Segoe UI", 15, "bold"),
            fg=FG_WHITE,
            bg=BG_DARK,
        ).pack(pady=14)

        top = tk.Frame(win, bg=BG_DARK)
        top.pack(fill="x", padx=20)

        tk.Label(top, text="Select Date:", fg=FG_WHITE, bg=BG_DARK,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))

        file_var = tk.StringVar(value=files[0])
        combo = ttk.Combobox(top, textvariable=file_var, values=files,
                             width=34, state="readonly")
        combo.pack(side="left")

        # Table
        cols = ("ID", "Name", "Time", "Status")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=BG_CARD,
            foreground=FG_WHITE,
            rowheight=26,
            fieldbackground=BG_CARD,
            font=("Segoe UI", 10),
        )
        style.configure("Treeview.Heading", background=BG_PANEL,
                        foreground=FG_WHITE, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT2)])

        tree_frame = tk.Frame(win, bg=BG_DARK)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=12)

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=160 if col == "Name" else 100, anchor="center")
        tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        count_lbl = tk.Label(win, text="", fg=GREEN, bg=BG_DARK,
                             font=("Segoe UI", 10))
        count_lbl.pack(pady=(0, 10))

        def load_records(*_):
            tree.delete(*tree.get_children())
            df = self.attendance_mgr.load_file(file_var.get())
            for _, row in df.iterrows():
                tag = "present" if str(row.get("Status", "")).lower() == "present" else "absent"
                tree.insert("", "end", values=tuple(row), tags=(tag,))
            tree.tag_configure("present", foreground=GREEN)
            tree.tag_configure("absent", foreground=ACCENT)
            count_lbl.config(text=f"Total records: {len(df)}")

        combo.bind("<<ComboboxSelected>>", load_records)
        load_records()

    def on_close(self):
        self._stop_attendance()
        self._stop_reg_camera()
        self.root.destroy()
