"""Core recognition engine for the Face Recognition Attendance System.

This package contains the framework-agnostic computer-vision logic, cleanly
separated from the Tkinter GUI:

    detectors.py    — face *detection* backends (Haar Cascade, YuNet CNN)
    recognizers.py  — face *recognition* backends (LBPH, SFace embeddings)
    pipeline.py     — orchestrates a detector + recognizer into one engine
    augmentation.py — training-time data augmentation
    dataset.py      — dataset loading / label parsing helpers

The GUI and the evaluation scripts both build on this package, so the same
code path that powers the live app is the one that gets benchmarked.
"""

import logging

from config import LOG_FILE

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure root logging to both console and file."""
    root = logging.getLogger()
    if root.handlers:                     # already configured
        return
    root.setLevel(level)

    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # read-only environment — console logging is enough
        pass


__all__ = ["configure_logging"]
