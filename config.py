"""Central configuration for the Face Recognition Attendance System.

Tweak these constants to adapt the system to different cameras,
lighting conditions, or class sizes — no logic changes required.
"""

# ── Recognition ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 80   # LBPH: lower = better match; <80 → recognized
MIN_FACE_PX          = 50   # ignore faces smaller than this (pixels)

# ── Registration ─────────────────────────────────────────────────────────────
CAPTURE_EVERY_N      = 4    # save 1 image every N frames (diversity throttle)
TARGET_IMAGES        = 50   # total face images captured per student

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"
TRAINER_DIR   = "trainer"
ATTENDANCE_DIR = "Attendance"
STUDENT_CSV   = "student_details.csv"
MODEL_PATH    = "trainer/trainer.yml"
CASCADE_PATH_RELATIVE = "haarcascade_frontalface_default.xml"

# ── Face dimensions ───────────────────────────────────────────────────────────
FACE_SIZE = (100, 100)      # all faces resized to this before training/prediction

# ── UI colours (GitHub dark-mode palette) ─────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#21262d"
ACCENT    = "#e94560"
ACCENT2   = "#1f6feb"
FG_WHITE  = "#f0f6fc"
FG_MUTED  = "#8b949e"
GREEN     = "#3fb950"
ORANGE    = "#d29922"
PURPLE    = "#533483"
