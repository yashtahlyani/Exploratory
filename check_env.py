#!/usr/bin/env python3
"""
Pre-flight environment check.
Run this once before launching the app to verify all dependencies and hardware.

    python check_env.py
"""

import sys
import importlib

# Windows consoles default to cp1252 and crash on ✓/✗/─ — force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DEPENDENCIES = [
    # (import_path,  pip_package,               label)
    ("cv2",          "opencv-contrib-python",   "OpenCV core"),
    ("cv2.face",     "opencv-contrib-python",   "OpenCV face module (contrib)"),
    ("PIL",          "Pillow",                  "Pillow (image I/O)"),
    ("numpy",        "numpy",                   "NumPy"),
    ("pandas",       "pandas",                  "pandas"),
    ("openpyxl",     "openpyxl",                "openpyxl (Excel export)"),
    ("matplotlib",   "matplotlib",              "matplotlib (analytics charts)"),
    ("sklearn",      "scikit-learn",            "scikit-learn (evaluation metrics)"),
    ("tkinter",      None,                      "Tkinter (GUI — built-in)"),
]


def _section(title: str):
    print(f"\n{title}")
    print("─" * 52)


def _check_python() -> bool:
    major, minor, micro = sys.version_info[:3]
    ok     = (major, minor) >= (3, 11)
    status = "✓" if ok else "✗"
    note   = "" if ok else "  ← Python 3.11+ required"
    _section("Python")
    print(f"  {status}  Python {major}.{minor}.{micro}{note}")
    return ok


def _check_imports() -> bool:
    _section("Libraries")
    ok = True
    for module, pip_pkg, label in _DEPENDENCIES:
        try:
            importlib.import_module(module)
            print(f"  ✓  {label}")
        except ImportError:
            install = f"  ← pip install {pip_pkg}" if pip_pkg else ""
            print(f"  ✗  {label}{install}")
            ok = False
    return ok


def _check_camera() -> bool:
    _section("Hardware")
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("  ✗  Webcam — not detected")
            return False
        ret, _ = cap.read()
        cap.release()
        if ret:
            print("  ✓  Webcam — detected and readable")
            return True
        print("  ✗  Webcam — opened but returned no frames")
        return False
    except Exception as exc:
        print(f"  ✗  Webcam check error: {exc}")
        return False


def _check_deep_models():
    """Optional — the deep-learning engine needs downloaded ONNX weights."""
    import os
    from config import YUNET_MODEL, SFACE_MODEL
    _section("Deep-learning engine (optional)")
    have_yunet = os.path.exists(YUNET_MODEL)
    have_sface = os.path.exists(SFACE_MODEL)
    print(f"  {'✓' if have_yunet else '·'}  YuNet detector  ({YUNET_MODEL})")
    print(f"  {'✓' if have_sface else '·'}  SFace recognizer ({SFACE_MODEL})")
    if not (have_yunet and have_sface):
        print("     → run  python download_models.py  to enable the deep engine")
        print("       (the classical engine works without these)")


def main() -> bool:
    print("\n  Face Recognition Attendance System — Environment Check")
    print("=" * 54)

    py_ok  = _check_python()
    lib_ok = _check_imports()
    cam_ok = _check_camera()
    _check_deep_models()

    print("\n" + "=" * 54)
    if py_ok and lib_ok and cam_ok:
        print("  All checks passed.  Run:  python main.py\n")
    else:
        print("  Fix the issues above, then re-run this script.\n")

    return py_ok and lib_ok and cam_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
