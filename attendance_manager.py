import os
import re
import pandas as pd
from datetime import datetime

from config import ATTENDANCE_DIR

_COLUMNS = ["ID", "Name", "Time", "Status"]


class AttendanceManager:
    def __init__(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        self._marked_today: set = set()

    # ── internal ──────────────────────────────────────────────────────────────

    def _today_file(self) -> str:
        date_str = datetime.now().strftime("%d-%m-%Y")
        return os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

    # ── marking ───────────────────────────────────────────────────────────────

    def mark_attendance(self, student_id: int, name: str) -> bool:
        """Mark attendance; returns True only if newly marked this session."""
        key = (student_id, datetime.now().strftime("%d-%m-%Y"))
        if key in self._marked_today:
            return False

        file_path = self._today_file()
        now       = datetime.now().strftime("%H:%M:%S")

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

    # ── querying ──────────────────────────────────────────────────────────────

    def get_today_records(self) -> pd.DataFrame:
        file_path = self._today_file()
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame(columns=_COLUMNS)

    def list_attendance_files(self) -> list[str]:
        files = [f for f in os.listdir(ATTENDANCE_DIR)
                 if f.endswith(".csv")]
        return sorted(files, reverse=True)

    def load_file(self, filename: str) -> pd.DataFrame:
        path = os.path.join(ATTENDANCE_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=_COLUMNS)

    # ── export ────────────────────────────────────────────────────────────────

    def export_to_excel(self, filename: str) -> str:
        """Export a single CSV to .xlsx; returns the output path."""
        df       = self.load_file(filename)
        out_name = filename.replace(".csv", ".xlsx")
        out_path = os.path.join(ATTENDANCE_DIR, out_name)
        self._write_xlsx({_date_from_filename(filename): df}, out_path)
        return out_path

    def export_all_to_excel(self) -> str:
        """Export every attendance CSV into one workbook (one sheet per date).

        Returns the output path.
        """
        files    = self.list_attendance_files()
        sheets   = {_date_from_filename(f): self.load_file(f) for f in files}
        out_path = os.path.join(ATTENDANCE_DIR, "Attendance_All.xlsx")
        self._write_xlsx(sheets, out_path)
        return out_path

    @staticmethod
    def _write_xlsx(sheets: dict[str, pd.DataFrame], out_path: str):
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                safe = sheet_name[:31]   # Excel sheet name limit
                df.to_excel(writer, index=False, sheet_name=safe)
                ws = writer.sheets[safe]
                for col in ws.columns:
                    max_len = max(
                        (len(str(cell.value or "")) for cell in col), default=0
                    )
                    ws.column_dimensions[col[0].column_letter].width = max_len + 4


def _date_from_filename(filename: str) -> str:
    """Extract 'DD-MM-YYYY' from 'Attendance_DD-MM-YYYY.csv'."""
    m = re.match(r"Attendance_(\d{2}-\d{2}-\d{4})\.csv$", filename)
    return m.group(1) if m else filename.replace(".csv", "")
