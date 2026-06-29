import os
import pandas as pd
from datetime import datetime


class AttendanceManager:
    def __init__(self):
        os.makedirs("Attendance", exist_ok=True)
        self._marked_today = set()

    def _today_file(self):
        date_str = datetime.now().strftime("%d-%m-%Y")
        return os.path.join("Attendance", f"Attendance_{date_str}.csv")

    def mark_attendance(self, student_id: int, name: str) -> bool:
        """Mark attendance for a student; returns True if newly marked."""
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
            df = pd.DataFrame(columns=["ID", "Name", "Time", "Status"])

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
        return pd.DataFrame(columns=["ID", "Name", "Time", "Status"])

    def list_attendance_files(self) -> list[str]:
        files = [f for f in os.listdir("Attendance") if f.endswith(".csv")]
        return sorted(files, reverse=True)

    def load_file(self, filename: str) -> pd.DataFrame:
        path = os.path.join("Attendance", filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=["ID", "Name", "Time", "Status"])
