"""Student Registry window — view enrolled students and remove records."""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

import pandas as pd

from config import (
    DATASET_DIR, STUDENT_CSV,
    BG_DARK, BG_PANEL, BG_CARD, FG_WHITE, FG_MUTED,
    GREEN, ACCENT, ACCENT2, ORANGE,
)


def _count_images_per_student() -> dict[int, int]:
    counts: dict[int, int] = {}
    if not os.path.exists(DATASET_DIR):
        return counts
    for fname in os.listdir(DATASET_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        parts = fname.split(".")
        if len(parts) >= 3:
            try:
                sid = int(parts[1])
                counts[sid] = counts.get(sid, 0) + 1
            except ValueError:
                continue
    return counts


class StudentRegistryWindow(tk.Toplevel):
    def __init__(self, parent, on_change: Callable | None = None):
        super().__init__(parent)
        self.title("Student Registry")
        self.geometry("660x480")
        self.configure(bg=BG_DARK)
        self.grab_set()
        self._on_change = on_change

        tk.Label(self, text="Student Registry",
                 font=("Segoe UI", 15, "bold"),
                 fg=FG_WHITE, bg=BG_DARK).pack(pady=12)

        info = tk.Label(
            self,
            text="Green = 50+ images captured  |  Orange = fewer images (retake recommended)",
            font=("Segoe UI", 9), fg=FG_MUTED, bg=BG_DARK)
        info.pack(pady=(0, 8))

        self._build_table()
        self._build_buttons()
        self._load()

    def _build_table(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Reg.Treeview",
                        background=BG_CARD, foreground=FG_WHITE, rowheight=28,
                        fieldbackground=BG_CARD, font=("Segoe UI", 10))
        style.configure("Reg.Treeview.Heading",
                        background=BG_PANEL, foreground=FG_WHITE,
                        font=("Segoe UI", 10, "bold"))
        style.map("Reg.Treeview", background=[("selected", ACCENT2)])

        wrap = tk.Frame(self, bg=BG_DARK)
        wrap.pack(fill="both", expand=True, padx=20, pady=(0, 6))

        sb = ttk.Scrollbar(wrap, orient="vertical")
        sb.pack(side="right", fill="y")

        self._tree = ttk.Treeview(
            wrap,
            columns=("ID", "Name", "Images", "Status"),
            show="headings",
            style="Reg.Treeview",
            yscrollcommand=sb.set,
        )
        sb.config(command=self._tree.yview)

        col_cfg = [
            ("ID",     80,  "center"),
            ("Name",   240, "w"),
            ("Images", 110, "center"),
            ("Status", 140, "center"),
        ]
        for col, width, anchor in col_cfg:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=width, anchor=anchor)

        self._tree.pack(side="left", fill="both", expand=True)

    def _build_buttons(self):
        self._total_lbl = tk.Label(self, text="", font=("Segoe UI", 10),
                                   fg=GREEN, bg=BG_DARK)
        self._total_lbl.pack(pady=(0, 4))

        row = tk.Frame(self, bg=BG_DARK)
        row.pack(pady=(0, 14))

        def _btn(text, color, cmd):
            return tk.Button(row, text=text, font=("Segoe UI", 10, "bold"),
                             bg=color, fg="white", relief="flat",
                             padx=14, pady=7, cursor="hand2", bd=0, command=cmd)

        _btn("Delete Selected", ACCENT,   self._delete_selected).pack(side="left", padx=5)
        _btn("Refresh",         ACCENT2,  self._load).pack(side="left", padx=5)
        _btn("Close",           FG_MUTED, self.destroy).pack(side="left", padx=5)

    def _load(self):
        for row in self._tree.get_children():
            self._tree.delete(row)

        if not os.path.exists(STUDENT_CSV):
            self._total_lbl.config(text="No students enrolled yet.")
            return

        df       = pd.read_csv(STUDENT_CSV)
        img_cnt  = _count_images_per_student()
        enrolled = len(df)

        for _, row in df.iterrows():
            sid    = int(row["ID"])
            count  = img_cnt.get(sid, 0)
            status = "Ready" if count >= 50 else ("Low images" if count > 0 else "No images")
            tag    = "ok" if count >= 50 else "low"
            self._tree.insert("", "end",
                              values=(sid, row["Name"], f"{count}", status),
                              tags=(tag,))

        self._tree.tag_configure("ok",  foreground=GREEN)
        self._tree.tag_configure("low", foreground=ORANGE)
        self._total_lbl.config(text=f"Total enrolled: {enrolled} student(s)")

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Click a student row to select them first.")
            return

        vals   = self._tree.item(sel[0])["values"]
        sid    = int(vals[0])
        name   = vals[1]
        images = vals[2]

        if not messagebox.askyesno(
            "Confirm deletion",
            f"Permanently delete {name} (ID {sid})?\n\n"
            f"• Their record will be removed from student_details.csv\n"
            f"• {images} face image(s) will be deleted from the dataset\n\n"
            "You will need to retrain the model afterwards.",
        ):
            return

        # remove from CSV
        if os.path.exists(STUDENT_CSV):
            df = pd.read_csv(STUDENT_CSV)
            df = df[df["ID"].astype(int) != sid]
            df.to_csv(STUDENT_CSV, index=False)

        # remove face images
        removed = 0
        if os.path.exists(DATASET_DIR):
            for fname in os.listdir(DATASET_DIR):
                parts = fname.split(".")
                if len(parts) >= 3:
                    try:
                        if int(parts[1]) == sid:
                            os.remove(os.path.join(DATASET_DIR, fname))
                            removed += 1
                    except (ValueError, OSError):
                        continue

        messagebox.showinfo(
            "Deleted",
            f"{name} removed successfully.\n"
            f"({removed} image(s) deleted)\n\n"
            "Remember to click  Train & Save  to update the recognition model.",
        )
        self._load()
        if self._on_change:
            self._on_change()
