import os
import pandas as pd
from datetime import datetime

from config import ATTENDANCE_DIR

_COLUMNS = ["ID", "Name", "Time", "Status"]


class AttendanceManager:
    def __init__(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        self._marked_today: set = set()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _today_file(self) -> str:
        date_str = datetime.now().strftime("%d-%m-%Y")
        return os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

    # ── public API ────────────────────────────────────────────────────────────

    def mark_attendance(self, student_id: int, name: str) -> bool:
        """Mark attendance for a student; returns True only if newly marked."""
        key = (student_id, datetime.now().strftime("%d-%m-%Y"))
        if key in self._marked_today:
            return False

        file_path = self._today_file()
        now = datetime.now().strftime("%H:%M:%S")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if student_id in df["ID"].values:
                self._marked_today.add(key)
                return False
        else:
            df = pd.DataFrame(columns=_COLUMNS)

        new_row = pd.DataFrame(
            {"ID": [student_id], "Name": [name], "Time": [now], "Status": ["Present"]}
        )
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)
        self._marked_today.add(key)
        return True

    def get_today_records(self) -> pd.DataFrame:
        file_path = self._today_file()
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame(columns=_COLUMNS)

    def list_attendance_files(self) -> list[str]:
        files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")]
        return sorted(files, reverse=True)

    def load_file(self, filename: str) -> pd.DataFrame:
        path = os.path.join(ATTENDANCE_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=_COLUMNS)

    def export_to_excel(self, filename: str) -> str:
        """Export a CSV attendance file to .xlsx; returns the output path."""
        df = self.load_file(filename)
        xlsx_name = filename.replace(".csv", ".xlsx")
        out_path  = os.path.join(ATTENDANCE_DIR, xlsx_name)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")
            ws = writer.sheets["Attendance"]

            # column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4

        return out_path
