# Keeper - face analysis.
#
# The whole-frame score misleads on portraits: a razor-sharp background with
# a soft face scores high, although the face is why the photo exists. Measuring
# the face region separately exposes that (real example: frame 99, face 60).
#
# OPTIONAL like rawpy: without the model file every function returns an
# empty report and the program works on without face information.
#
# There is no eye-open check on purpose: YuNet only gives the eye centres,
# and a shape-based estimate tested on 791 real faces mostly found eyebrows or
# shadows and called closed eyes open. A misleading signal is worse than none.

import io
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from core import noise, raw_loader, sharpness

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / config.FACE_MODEL_FILE


def model_available():
    """The model file is the only requirement of the face pass."""
    return MODEL_PATH.is_file()


# One detector PER THREAD. The face pass runs on a thread pool, and a shared
# FaceDetectorYN keeps its input size and buffers inside the instance:
# measured on 24 frames x 28 threads, a shared one gave 98/144 wrong face
# counts and 88 exceptions, per-thread instances 0/0 (0.55 MB each).
_thread_local = threading.local()


def _detector_for(width, height):
    """The CALLING thread's detector, set to the given input size."""
    if not model_available():
        return None

    detector = getattr(_thread_local, "detector", None)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH), "", (width, height),
            config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD, config.FACE_TOP_K,
        )
        _thread_local.detector = detector
        _thread_local.size = (width, height)

    if _thread_local.size != (width, height):
        detector.setInputSize((width, height))
        _thread_local.size = (width, height)

    return detector


@dataclass
class Face:
    """One detected face."""

    box: tuple            # (x, y, width, height)
    right_eye: tuple      # (x, y) - the eye on the RIGHT side of the image
    left_eye: tuple
    confidence: float
    # further YuNet landmarks; head pose is computed from them
    nose: tuple = None
    mouth_right: tuple = None
    mouth_left: tuple = None
    # filled in by the analysis
    sharpness: float = None   # 0-100, measured on the face region
    yaw: float = None         # head turn, see head_pose()
    roll: float = None        # head tilt in degrees

    def __post_init__(self):
        # pose is a pure function of the landmarks, so it is always consistent
        # wherever the face came from (detector, test, saved data)
        if self.yaw is None and self.roll is None:
            self.yaw, self.roll = head_pose(self)

    @property
    def area(self):
        return self.box[2] * self.box[3]

    @property
    def faces_camera(self):
        """
        Looks towards the camera (None if unknown). Lenient on purpose: the
        question is whether the face is visible; a three-quarter profile is a
        fine portrait.
        """
        if self.yaw is None:
            return None
        return abs(self.yaw) <= config.FACE_YAW_FRONTAL


@dataclass
class FaceReport:
    """Face analysis of one shot."""

    faces: list = field(default_factory=list)
    error: str = None

    @property
    def count(self):
        return len(self.faces)

    @property
    def main(self):
        """The LARGEST face - the subject, not a stranger in the background."""
        if not self.faces:
            return None
        return max(self.faces, key=lambda face: face.area)


def detect(bgr):
    """Faces on a BGR image. Never raises; without the model an empty report."""
    if not model_available():
        return FaceReport()

    height, width = bgr.shape[:2]
    try:
        _, rows = _detector_for(width, height).detect(bgr)
    except Exception as exc:
        return FaceReport(error=str(exc))

    if rows is None:
        return FaceReport()

    # YuNet row: x, y, w, h, right eye, left eye, nose tip, right and left
    # mouth corner, confidence
    return FaceReport(faces=[
        Face(
            box=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
            right_eye=(float(r[4]), float(r[5])),
            left_eye=(float(r[6]), float(r[7])),
            nose=(float(r[8]), float(r[9])),
            mouth_right=(float(r[10]), float(r[11])),
            mouth_left=(float(r[12]), float(r[13])),
            confidence=float(r[-1]),
        )
        for r in rows
    ])


def head_pose(face):
    """
    (yaw, roll) from the five landmarks; either may be None.

      yaw   nose-tip offset from the eye midpoint along the eye axis, divided
            by the eye distance (~ -1..+1, 0 = frontal; not degrees). The
            projection makes it independent of tilt and face size.
      roll  head tilt in degrees from the eye line.

    Pitch is not estimated: the vertical nose position depends on face shape
    as much as on pose. That is also why pose only affects ranking.
    """
    if face.nose is None:
        return None, None

    right = np.array(face.right_eye, dtype=np.float64)
    left = np.array(face.left_eye, dtype=np.float64)
    nose = np.array(face.nose, dtype=np.float64)

    axis = left - right
    distance = float(np.hypot(*axis))
    # eyes closer than 2 px: the face is too small or the detection slipped
    if distance < 2.0:
        return None, None

    # image y grows downwards, hence the minus: positive = counter-clockwise
    roll = float(np.degrees(np.arctan2(-axis[1], axis[0])))
    middle = (right + left) / 2.0
    yaw = float(np.dot(nose - middle, axis / distance) / distance)
    return yaw, roll


def face_sharpness(gray, face, sigma=None):
    """
    Sharpness of the face region only, with the SAME function as the whole
    frame (so the two compare). Noise comes from the whole frame: a face crop
    is mostly texture and too small for a noise estimate. None if the face is
    too small to measure.
    """
    height, width = gray.shape[:2]
    x, y, w, h = face.box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))

    if x1 - x0 < config.FACE_MIN_SIZE or y1 - y0 < config.FACE_MIN_SIZE:
        return None

    score, _, _, _ = sharpness.score_parts(gray[y0:y1, x0:x1], sigma)
    return score


def _resize(bgr, longer_side):
    height, width = bgr.shape[:2]
    longer = max(height, width)
    if longer <= longer_side:
        return bgr
    ratio = longer_side / longer
    return cv2.resize(
        bgr, (max(1, round(width * ratio)), max(1, round(height * ratio))),
        interpolation=cv2.INTER_AREA,
    )


def load_bgr_for_faces(path, longer_side=None):
    """Colour image scaled to longer_side; RAW files use the embedded preview."""
    longer_side = longer_side or config.FACE_WORK_RESOLUTION
    path = Path(path)

    if raw_loader.is_raw(path):
        data = raw_loader.preview_jpeg_bytes(path)
        if data is None:
            rgb = raw_loader.demosaic_rgb(path)
            return _resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), longer_side)
        source = io.BytesIO(data)
    else:
        source = path

    with Image.open(source) as img:
        img.draft("RGB", (longer_side, longer_side))
        img = ImageOps.exif_transpose(img)
        rgb = np.array(img.convert("RGB"))

    return _resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), longer_side)


def analyse(bgr):
    """Detection plus face sharpness on an already loaded image."""
    report = detect(bgr)
    if report.faces:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sigma = noise.estimate_sigma(gray)  # whole-frame noise for every face
        for face in report.faces:
            face.sharpness = face_sharpness(gray, face, sigma)
    return report


def analyse_path(path):
    """
    Face analysis of a file at the sharpness work resolution (so face and
    frame scores share one pixel scale). Never raises.
    """
    if not model_available():
        return FaceReport()
    try:
        return analyse(load_bgr_for_faces(path, config.WORK_RESOLUTION))
    except Exception as exc:
        return FaceReport(error=str(exc))
