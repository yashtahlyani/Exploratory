"""Analytics window — daily attendance trends and per-student statistics."""

import os
import re
import tkinter as tk
from tkinter import ttk
from datetime import datetime

import pandas as pd

from config import (
    ATTENDANCE_DIR, STUDENT_CSV,
    BG_DARK, BG_PANEL, FG_WHITE, FG_MUTED,
    GREEN, ACCENT, ACCENT2, ORANGE,
)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.ticker as mticker
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

_MPL_RC = {
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_PANEL,
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  FG_WHITE,
    "xtick.color":      FG_MUTED,
    "ytick.color":      FG_MUTED,
    "text.color":       FG_WHITE,
    "grid.color":       "#30363d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
}


def _load_all_sessions() -> dict[str, pd.DataFrame]:
    sessions: dict[str, pd.DataFrame] = {}
    if not os.path.exists(ATTENDANCE_DIR):
        return sessions
    for fname in sorted(os.listdir(ATTENDANCE_DIR)):
        if not fname.endswith(".csv"):
            continue
        m = re.match(r"Attendance_(\d{2}-\d{2}-\d{4})\.csv$", fname)
        if not m:
            continue
        try:
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, fname))
            sessions[m.group(1)] = df
        except Exception:
            continue
    return sessions


def _load_students() -> pd.DataFrame:
    if os.path.exists(STUDENT_CSV):
        return pd.read_csv(STUDENT_CSV)
    return pd.DataFrame(columns=["ID", "Name"])


class AnalyticsWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Attendance Analytics")
        self.geometry("920x620")
        self.configure(bg=BG_DARK)
        self.grab_set()

        if _HAS_MATPLOTLIB:
            matplotlib.rcParams.update(_MPL_RC)

        self._sessions = _load_all_sessions()
        self._students = _load_students()

        tk.Label(self, text="Attendance Analytics",
                 font=("Segoe UI", 15, "bold"),
                 fg=FG_WHITE, bg=BG_DARK).pack(pady=12)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BG_PANEL, foreground=FG_MUTED,
                        padding=[14, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT2)],
                  foreground=[("selected", FG_WHITE)])

        self._build_daily_tab(nb)
        self._build_student_tab(nb)
        self._build_summary_tab(nb)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _no_data_label(self, parent, msg="No attendance records found."):
        tk.Label(parent, text=msg, fg=FG_MUTED, bg=BG_DARK,
                 font=("Segoe UI", 11)).pack(expand=True)

    def _no_matplotlib_label(self, parent):
        tk.Label(parent,
                 text="matplotlib not installed.\n\nRun:  pip install matplotlib",
                 fg=ORANGE, bg=BG_DARK, font=("Segoe UI", 11)).pack(expand=True)

    def _embed(self, parent, fig):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    # ── daily trend tab ───────────────────────────────────────────────────────

    def _build_daily_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(frame, text="  Daily Trend  ")

        if not _HAS_MATPLOTLIB:
            self._no_matplotlib_label(frame)
            return
        if not self._sessions:
            self._no_data_label(frame)
            return

        dates  = sorted(self._sessions.keys())[-14:]
        counts = [len(self._sessions[d]) for d in dates]
        disp   = []
        for d in dates:
            try:
                disp.append(datetime.strptime(d, "%d-%m-%Y").strftime("%d %b"))
            except ValueError:
                disp.append(d)

        fig = Figure(figsize=(8.4, 4.4), dpi=95)
        ax  = fig.add_subplot(111)

        bars = ax.bar(range(len(disp)), counts, color=ACCENT2, width=0.6, zorder=2)
        ax.set_xticks(range(len(disp)))
        ax.set_xticklabels(disp, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Date", labelpad=8)
        ax.set_ylabel("Students Present", labelpad=8)
        ax.set_title("Daily Attendance — Last 14 Sessions", pad=14, fontsize=11)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", zorder=1)

        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom",
                    fontsize=8, color=FG_WHITE)

        fig.tight_layout(pad=1.6)
        self._embed(frame, fig)

    # ── per-student tab ───────────────────────────────────────────────────────

    def _build_student_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(frame, text="  Per Student  ")

        if not _HAS_MATPLOTLIB:
            self._no_matplotlib_label(frame)
            return
        if self._students.empty or not self._sessions:
            self._no_data_label(frame, "Need enrolled students + attendance records.")
            return

        total_sessions = len(self._sessions)
        present_counts: dict[int, int] = {}
        for df in self._sessions.values():
            for sid in df["ID"].astype(int):
                present_counts[sid] = present_counts.get(sid, 0) + 1

        names, pcts = [], []
        for _, row in self._students.iterrows():
            sid = int(row["ID"])
            names.append(str(row["Name"]))
            pcts.append(round(present_counts.get(sid, 0) / total_sessions * 100, 1))

        paired = sorted(zip(pcts, names), reverse=True)
        if paired:
            pcts_s, names_s = zip(*paired)
        else:
            pcts_s, names_s = [], []

        bar_colors = [GREEN if p >= 75 else ORANGE if p >= 50 else ACCENT
                      for p in pcts_s]

        fig_h = max(3.6, len(names_s) * 0.58 + 1.6)
        fig   = Figure(figsize=(8.4, fig_h), dpi=95)
        ax    = fig.add_subplot(111)

        bars = ax.barh(list(names_s), list(pcts_s),
                       color=bar_colors, height=0.55, zorder=2)
        ax.set_xlabel("Attendance %", labelpad=8)
        ax.set_title(
            f"Attendance % per Student  ({total_sessions} total sessions)",
            pad=14, fontsize=11)
        ax.set_xlim(0, 115)
        ax.grid(axis="x", zorder=1)
        ax.axvline(75, color=FG_MUTED, linestyle=":", linewidth=1.2,
                   label="75% threshold")
        ax.legend(fontsize=8, loc="lower right")

        for bar, val in zip(bars, pcts_s):
            ax.text(bar.get_width() + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val}%", va="center", fontsize=8, color=FG_WHITE)

        fig.tight_layout(pad=1.6)
        self._embed(frame, fig)

    # ── summary tab ───────────────────────────────────────────────────────────

    def _build_summary_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(frame, text="  Summary  ")

        if not self._sessions:
            self._no_data_label(frame)
            return

        total_sessions  = len(self._sessions)
        total_enrolled  = len(self._students)
        all_counts      = [len(df) for df in self._sessions.values()]
        avg_present     = round(sum(all_counts) / len(all_counts), 1) if all_counts else 0
        avg_pct         = round(avg_present / total_enrolled * 100, 1) if total_enrolled else 0

        best_date   = max(self._sessions, key=lambda d: len(self._sessions[d]))
        worst_date  = min(self._sessions, key=lambda d: len(self._sessions[d]))

        name_map: dict[int, str] = {}
        if not self._students.empty:
            name_map = dict(zip(self._students["ID"].astype(int),
                                self._students["Name"].astype(str)))
        present_counts: dict[int, int] = {}
        for df in self._sessions.values():
            for sid in df["ID"].astype(int):
                present_counts[sid] = present_counts.get(sid, 0) + 1

        top_attender = "—"
        if present_counts:
            top_id       = max(present_counts, key=present_counts.get)
            top_attender = name_map.get(top_id, f"ID {top_id}")

        metrics = [
            ("Total Sessions",         str(total_sessions),                                ACCENT2),
            ("Enrolled Students",      str(total_enrolled),                                ACCENT2),
            ("Avg Present / Session",  f"{avg_present}",                                  GREEN),
            ("Avg Attendance Rate",    f"{avg_pct}%",                                     GREEN),
            ("Best Session",           f"{best_date}  ({len(self._sessions[best_date])})", GREEN),
            ("Lowest Session",         f"{worst_date}  ({len(self._sessions[worst_date])})", ORANGE),
            ("Perfect Attender",       top_attender,                                       GREEN),
        ]

        outer = tk.Frame(frame, bg=BG_DARK)
        outer.pack(expand=True, pady=20)

        for i, (label, value, color) in enumerate(metrics):
            card = tk.Frame(outer, bg=BG_PANEL, highlightthickness=1,
                            highlightbackground="#30363d")
            card.grid(row=i // 2, column=i % 2,
                      padx=16, pady=10, sticky="nsew", ipadx=20, ipady=12)
            tk.Label(card, text=label, font=("Segoe UI", 9),
                     fg=FG_MUTED, bg=BG_PANEL).pack(anchor="w")
            tk.Label(card, text=value, font=("Segoe UI", 17, "bold"),
                     fg=color, bg=BG_PANEL).pack(anchor="w")

        for col in range(2):
            outer.columnconfigure(col, weight=1)
