"""Tests for AttendanceManager — marking, dedup, and Excel export.

AttendanceManager writes to the relative ``Attendance/`` directory, so each
test runs inside a temporary working directory via monkeypatch.chdir.
"""

import os

import pytest

from attendance_manager import AttendanceManager


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return AttendanceManager()


def test_mark_returns_true_first_time(mgr):
    assert mgr.mark_attendance(1, "Alice") is True


def test_mark_is_idempotent_per_day(mgr):
    assert mgr.mark_attendance(1, "Alice") is True
    assert mgr.mark_attendance(1, "Alice") is False   # already marked
    records = mgr.get_today_records()
    assert len(records) == 1


def test_multiple_students_recorded(mgr):
    mgr.mark_attendance(1, "Alice")
    mgr.mark_attendance(2, "Bob")
    df = mgr.get_today_records()
    assert set(df["ID"]) == {1, 2}
    assert set(df["Status"]) == {"Present"}


def test_dedup_survives_new_manager_instance(mgr, tmp_path, monkeypatch):
    mgr.mark_attendance(1, "Alice")
    # a fresh manager (e.g. app restart) must read the existing CSV
    monkeypatch.chdir(tmp_path)
    fresh = AttendanceManager()
    assert fresh.mark_attendance(1, "Alice") is False


def test_export_to_excel_creates_file(mgr):
    mgr.mark_attendance(1, "Alice")
    files = mgr.list_attendance_files()
    assert files, "expected one CSV after marking"
    out = mgr.export_to_excel(files[0])
    assert os.path.exists(out)
    assert out.endswith(".xlsx")


def test_export_all_to_excel(mgr):
    mgr.mark_attendance(1, "Alice")
    out = mgr.export_all_to_excel()
    assert os.path.exists(out)
    assert out.endswith("Attendance_All.xlsx")
