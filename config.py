# Keeper - every tunable value in one place.
#
# The numbers below were calibrated on real photos (68 sunny ISO-100 frames
# plus blurred variants, 136 frames for tone). The long-form calibration notes
# from the original (Hungarian) development are preserved in the git history of
# this file (baseline commit); tests/test_calibration.py guards the results.

import os

# ---------------------------------------------------------------------------
# sharpness measurement
# ---------------------------------------------------------------------------

# Every image is scaled to this longer side before measuring, so the score
# does not depend on the camera resolution.
WORK_RESOLUTION = 1024

# Calibration ranges of the three metrics (contrast-normalised and with the
# noise contribution removed - see core/sharpness.py metrics()).
# LO maps to 0 points, HI to 100. HI = 1.15 x the sharpest real frame,
# LO puts the 5th percentile of real frames at 32 points.
LAPLACIAN_LO = 0.0957
LAPLACIAN_HI = 0.5029
TENENGRAD_LO = 1.1430
TENENGRAD_HI = 4.8454
BRENNER_LO = 0.0500
BRENNER_HI = 0.1732

# Noise contribution to each metric, as a multiple of sigma^2. These follow
# from the filter coefficients (sum of squares): Laplacian 20, Sobel x+y 24,
# Brenner 2 - verified on synthetic white noise.
NOISE_LAPLACIAN_GAIN = 20.0
NOISE_TENENGRAD_GAIN = 24.0
NOISE_BRENNER_GAIN = 2.0

# Lower bound of the contrast normaliser (std on a 0-255 scale); protects
# nearly flat images from division by ~0.
CONTRAST_FLOOR = 8.0

# Weights of the three metrics; must sum to exactly 1.0 (checked at import).
WEIGHT_LAPLACIAN = 1.0 / 3.0
WEIGHT_TENENGRAD = 1.0 / 3.0
WEIGHT_BRENNER = 1.0 / 3.0

# Default threshold: frames scoring below it are rejects. On the reference
# set every visually soft frame scored <= 47 and every sharp one >= 63.
DEFAULT_THRESHOLD = 50

# Pillow refuses images above ~179 MP as a "decompression bomb". Panoramas can
# be larger, so the limit is raised deliberately (measuring uses draft()).
MAX_IMAGE_PIXELS = 400_000_000

# ---------------------------------------------------------------------------
# expectation from the shooting conditions (EXIF)
# ---------------------------------------------------------------------------
# Easy conditions (low ISO, fast shutter) make us stricter, hard ones more
# lenient. STRICT_PENALTY + LENIENT_BONUS (12) stays below the 15.4-point gap
# between the visually soft and sharp groups, so EXIF alone can never flip a
# clearly sharp or clearly soft frame.

STRICT_SHAKE_STOPS = 2.0     # >= 2 stops faster than 1/focal: no excuse
LENIENT_SHAKE_STOPS = 2.0    # >= 2 stops slower: softness is forgivable
STRICT_MAX_ISO = 400         # up to here the light counts as "good"
LENIENT_FULL_ISO = 3200      # full leniency from here (log2 ramp between)
STRICT_PENALTY = 4.0
LENIENT_BONUS = 8.0

# Header-only hints shown on the card. They never reject anything.
WARN_SHAKE_STOPS = -1.0
WARN_HIGH_ISO = 3200

# ---------------------------------------------------------------------------
# files and folders
# ---------------------------------------------------------------------------

JPEG_EXTENSIONS = (".jpg", ".jpeg")
# Only formats that were actually tested; LibRaw reads many more.
RAW_EXTENSIONS = (".arw",)
IMAGE_EXTENSIONS = JPEG_EXTENSIONS + RAW_EXTENSIONS

# A RAW and a JPEG with the same stem are ONE shot and always move together.
PAIR_RAW_WITH_JPEG = True

FOLDER_REJECTS = "rejected"
FOLDER_DUPLICATES = "duplicates"
FOLDER_CORRECTED = "corrected"

# Folder names written by versions before 2026-10 (Hungarian). Undo still
# accepts them, so older logs remain undoable.
LEGACY_FOLDERS = {
    "selejt": FOLDER_REJECTS,
    "ismetlodes": FOLDER_DUPLICATES,
    "javitott": FOLDER_CORRECTED,
}

# The undo log lives in the processed folder.
LOG_FILENAME = ".keeper_log.json"

# Write-ahead logging: the log is rewritten (atomically) before every batch
# of this many shots is moved, so an interrupted run is still undoable.
LOG_BATCH = 32

# Paths at least this long get the Windows extended-length prefix
# (MAX_PATH is 260 characters).
LONG_PATH_LIMIT = 240

# ---------------------------------------------------------------------------
# faces
# ---------------------------------------------------------------------------

# YuNet model in assets/ (230 kB, MIT). Without it the face pass is skipped.
FACE_MODEL_FILE = "face_detection_yunet.onnx"
# Same scale as the sharpness measurement so face and frame scores compare.
FACE_WORK_RESOLUTION = WORK_RESOLUTION
FACE_SCORE_THRESHOLD = 0.8   # YuNet defaults
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 50
# Smaller faces are not measured (measured: 32 px gave 96 % coverage).
FACE_MIN_SIZE = 32
# Below this face score the card warns that the face is soft.
FACE_SOFT_THRESHOLD = 45.0
# Head turn limit for "faces the camera". yaw ~ 0.5 * tan(angle), so 0.35 is
# roughly a 35 degree turn - a three-quarter profile still counts.
FACE_YAW_FRONTAL = 0.35

# ---------------------------------------------------------------------------
# series (bursts)
# ---------------------------------------------------------------------------

# Consecutive frames at most this far apart (EXIF capture time) can form a
# series. Frames without an EXIF capture time never form a series.
SERIES_MAX_GAP_SECONDS = 6.0
# Max Hamming distance of the 64-bit perceptual hashes. Measured: same scene
# 0-10, different scene 22-38, so 14 sits safely in the gap.
SERIES_MAX_DISTANCE = 14
SERIES_MIN_SIZE = 2
# How many frames to keep per series when filtering duplicates.
SERIES_KEEP_DEFAULT = 1
SERIES_KEEP_MAX = 3

# Ranking inside a series only (never used for rejecting):
RANK_FACE_WEIGHT = 0.5      # share of the main face sharpness in the rank
RANK_FRONTAL_BONUS = 10.0   # bonus when the main face looks at the camera

# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------

# Noise is measured on a native-resolution centre crop (downscaling would
# average the noise away).
NOISE_CROP = 512
# Measured sigma (0-255 scale): below LOW nothing is filtered, above HIGH the
# strongest filtering is used.
NOISE_SIGMA_LOW = 1.5
NOISE_SIGMA_HIGH = 10.0
# ISO sanity cap: at ISO 100 a measured sigma above 2.0 is not believed (the
# estimator mistook fine detail for noise on 1 of 68 frames). The cap grows
# with sqrt(ISO) - photon shot noise.
ISO_SIGMA_CAP_BASE = 2.0
ISO_REFERENCE = 100.0
ISO_SIGMA_CAP_EXPONENT = 0.5

# ---------------------------------------------------------------------------
# denoising and sharpening
# ---------------------------------------------------------------------------

# NLM strength range. Kept low on purpose: h=3..5 removes visible grain while
# thin lines survive; above ~7 detail starts to smear.
DENOISE_H_MIN = 1.5
DENOISE_H_MAX = 5.0
# Chroma noise carries no detail, so it is filtered twice as hard.
DENOISE_COLOR_FACTOR = 2.0
# OpenCV NLM defaults (the search window costs quadratically).
DENOISE_TEMPLATE_SIZE = 7
DENOISE_SEARCH_SIZE = 21

# Sharpening follows the score: nothing below FLOOR (unrecoverable), the most
# at PEAK (slightly soft), only a little at CEILING (already sharp - halos).
SHARPEN_SCORE_FLOOR = 40.0
SHARPEN_SCORE_PEAK = 70.0
SHARPEN_SCORE_CEILING = 100.0
SHARPEN_MAX_AMOUNT = 0.8
SHARPEN_MIN_AMOUNT = 0.15
SHARPEN_RADIUS = 1.2
# Unsharp-mask threshold: the fixed floor, raised to GUARD x residual noise
# so sharpening never re-amplifies the grain that denoising removed.
SHARPEN_THRESHOLD = 3.0
SHARPEN_NOISE_GUARD = 3.0

# Large images are denoised and sharpened in tiles to bound memory use (and
# to allow cancelling between tiles). The overlap covers the NLM search window.
ENHANCE_TILE = 1024
ENHANCE_TILE_OVERLAP = 32

# ---------------------------------------------------------------------------
# user strength sliders (percent of the automatic strength)
# ---------------------------------------------------------------------------

ENHANCE_SCALE_DEFAULT = 100
ENHANCE_SCALE_MAX = 300
# White balance is OFF by default: on the reference set it neutralised warm
# subjects (brick, skin) and blue skies more often than it fixed real casts.
WB_SCALE_DEFAULT = 0
# Colour / local contrast is ON by default: it only acts on flat or dull
# frames and cannot clip highlights.
COLOR_SCALE_DEFAULT = 100
# Hard ceilings that no slider can exceed.
DENOISE_H_LIMIT = 15.0
SHARPEN_AMOUNT_LIMIT = 2.0

# ---------------------------------------------------------------------------
# tone and colour
# ---------------------------------------------------------------------------

# Tone statistics (white balance, exposure, contrast span, saturation) are
# measured on a copy scaled to this longer side. The preview and the saved
# copy use the same statistics, so the preview shows what will be saved.
TONE_STATS_RESOLUTION = 1024

WB_MINKOWSKI_P = 6          # shades-of-grey exponent
WB_MIN_DEVIATION = 0.03     # smaller casts are estimator noise
WB_GAIN_LIMIT = 1.18        # never push a channel further (keeps sunsets warm)

# Acceptable band for the brightness median; outside it we only pull back to
# the nearest edge of the band (a moody dark frame is not turned into noon).
EXPOSURE_TARGET_LOW = 70.0
EXPOSURE_TARGET_HIGH = 175.0
EXPOSURE_GAMMA_LIMIT = 1.8

# Local contrast (CLAHE on L*). Span = 95th - 5th percentile of lightness.
CONTRAST_TARGET_SPAN = 150.0   # at or above: contrasty enough, untouched
CONTRAST_FLAT_SPAN = 60.0      # full strength at this span
CONTRAST_MIN_SPAN = 5.0        # flat frame (lens cap): nothing to stretch
CLAHE_CLIP_MAX = 2.0           # hard ceiling (OpenCV default 40 is brutal)
CLAHE_CLIP_MIN = 0.6           # below this CLAHE does nothing visible
# Measured noise model: sigma_out = (1 + GAIN * clip) * sigma_in; the clip is
# limited so the output noise stays under the ceiling.
CLAHE_NOISE_GAIN = 0.7
CLAHE_NOISE_CEILING = 3.0
CLAHE_GRID = (8, 8)

# Saturation (HSV S median of usable pixels). Only dull outliers are lifted.
SATURATION_TARGET = 85.0
SATURATION_SCALE_LIMIT = 1.35
SATURATION_MIN_VALUE = 25
SATURATION_MAX_VALUE = 245

# ---------------------------------------------------------------------------
# learned denoiser (SCUNet) - OPTIONAL: needs PyTorch + CUDA + weights
# ---------------------------------------------------------------------------

LEARNED_MODEL_DIR = "models"
LEARNED_MODEL_FILE = "scunet_color_real_psnr.pth"
LEARNED_MODEL_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
)
LEARNED_MODEL_SHA256 = "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e"
# Network shape expected by the weights - not tunable.
LEARNED_BLOCKS = (4, 4, 4, 4, 4, 4, 4)
LEARNED_DIM = 64
# Tile size: a 42 MP frame in one pass needs ~10 GB VRAM; 1024 px tiles peak
# at ~3.2 GB. The overlap hides tile seams.
LEARNED_TILE = 1024
LEARNED_TILE_OVERLAP = 32

# ---------------------------------------------------------------------------
# preview and saving
# ---------------------------------------------------------------------------

# Preview crop at native resolution (denoising it takes ~0.5 s).
PREVIEW_CROP = 480
PREVIEW_DEBOUNCE_MS = 220

# Corrected copies go to the corrected/ subfolder; originals are never touched.
CORRECTED_JPEG_QUALITY = 95
CORRECTED_KEEP_EXIF = True
# Quick mode "Save as..." default name: <stem>_keeper.jpg next to the source.
QUICK_SAVE_SUFFIX = "_keeper"

# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------

# OpenCV and JPEG decoding release the GIL, so threads scale well.
WORKER_THREADS = os.cpu_count() or 4
