#!/usr/bin/env python3
"""Download the deep-learning model weights for the "deep" recognition engine.

Fetches two small ONNX files from the official OpenCV Zoo into ``models/``:

    • YuNet  (~340 KB)  — CNN face detector
    • SFace  (~37 MB)   — CNN face-embedding model

The classical (Haar + LBPH) engine needs none of this and works out of the
box; run this script only if you want the deep-learning engine.

    python download_models.py
"""

import os
import sys
import urllib.request

from config import MODELS_DIR, YUNET_MODEL, SFACE_MODEL, YUNET_URL, SFACE_URL

# Windows consoles default to cp1252 and crash on ✓/✗/─ — force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DOWNLOADS = [
    ("YuNet face detector", YUNET_URL, YUNET_MODEL),
    ("SFace recognizer",    SFACE_URL, SFACE_MODEL),
]


def _progress(label: str):
    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_048_576
        sys.stdout.write(f"\r  {label:22}  {pct:3d}%  ({mb:5.1f} MB)")
        sys.stdout.flush()
    return hook


def download_one(label: str, url: str, dest: str) -> bool:
    if os.path.exists(dest):
        print(f"  {label:22}  already present  ✓")
        return True
    try:
        urllib.request.urlretrieve(url, dest, _progress(label))
        print("  ✓")
        return True
    except Exception as exc:
        print(f"\n  ✗  Failed to download {label}: {exc}")
        if os.path.exists(dest):
            os.remove(dest)        # don't leave a truncated file behind
        return False


def main() -> int:
    print("\n  Downloading deep-learning models (OpenCV Zoo)")
    print("=" * 52)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ok = all(download_one(label, url, dest) for label, url, dest in _DOWNLOADS)

    print("=" * 52)
    if ok:
        print("  All models ready. The 'Deep Learning' engine is now")
        print("  selectable in the app's toolbar.\n")
        return 0
    print("  Some downloads failed. Check your internet connection and")
    print("  re-run. The classical engine still works without these.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
