"""Central configuration for the Face Recognition Attendance System.

Every tuneable constant lives here so the system can be adapted to a new
camera, lighting condition, class size, or recognition engine without
touching any logic. Imported by the GUI, the core pipeline, and the
evaluation scripts alike.
"""

# ── Recognition engines ───────────────────────────────────────────────────────
# "classical" = Haar Cascade detector + LBPH recognizer  (no downloads needed)
# "deep"      = YuNet CNN detector  + SFace CNN embeddings (run download_models.py)
DEFAULT_ENGINE = "classical"

# ── Classical engine (LBPH) ───────────────────────────────────────────────────
# LBPH returns a *distance*: lower = better match. A face is recognized when its
# distance is below this threshold. Exposed live via the GUI slider (50–100).
LBPH_THRESHOLD       = 80
LBPH_THRESHOLD_RANGE = (50, 100)

# Back-compat alias (older imports referenced CONFIDENCE_THRESHOLD)
CONFIDENCE_THRESHOLD = LBPH_THRESHOLD

# ── Deep engine (SFace) ───────────────────────────────────────────────────────
# SFace returns a *cosine similarity* in roughly [0, 1]: higher = better match.
# Stored here as a percentage (0–100). OpenCV's recommended cutoff is ~0.363.
SFACE_THRESHOLD       = 36
SFACE_THRESHOLD_RANGE = (20, 60)
YUNET_SCORE_THRESHOLD = 0.9   # YuNet detection confidence cutoff

# ── Detection ─────────────────────────────────────────────────────────────────
MIN_FACE_PX = 50    # ignore faces smaller than this (pixels)

# ── Registration ──────────────────────────────────────────────────────────────
CAPTURE_EVERY_N = 4     # save 1 image every N frames (diversity throttle)
TARGET_IMAGES   = 50    # total face images captured per student

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"
TRAINER_DIR    = "trainer"
ATTENDANCE_DIR = "Attendance"
MODELS_DIR     = "models"
ASSETS_DIR     = "assets"
STUDENT_CSV    = "student_details.csv"
LOG_FILE       = "attendance.log"

# model artifacts
MODEL_PATH     = "trainer/trainer.yml"            # LBPH model
SFACE_GALLERY  = "trainer/sface_gallery.npz"      # SFace identity embeddings
YUNET_MODEL    = "models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL    = "models/face_recognition_sface_2021dec.onnx"

CASCADE_PATH_RELATIVE = "haarcascade_frontalface_default.xml"

# Official OpenCV Zoo download URLs (used by download_models.py)
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/"
             "models/face_recognition_sface/face_recognition_sface_2021dec.onnx")

# ── Face dimensions ───────────────────────────────────────────────────────────
FACE_SIZE       = (100, 100)    # LBPH training/prediction size
SFACE_ALIGN_SIZE = (112, 112)   # SFace expects 112×112 aligned crops

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
VIOLET    = "#8957e5"
