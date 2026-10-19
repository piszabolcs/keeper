# Keeper - face analysis. Most of the value (face sharpness, ranking, pose,
# graceful degradation) is tested with hand-made faces; only detection itself
# needs the model.

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from core import faces, series, sharpness


def test_without_the_model_everything_is_empty_not_broken(monkeypatch, tmp_path):
    monkeypatch.setattr(faces, "model_available", lambda: False)

    report = faces.detect(np.zeros((100, 100, 3), np.uint8))

    assert report.count == 0 and report.main is None and report.error is None
    assert faces.analyse_path(tmp_path / "missing.jpg").count == 0


def test_a_broken_file_gives_an_error_not_an_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(faces, "model_available", lambda: True)
    path = tmp_path / "junk.jpg"
    path.write_bytes(b"not an image")

    report = faces.analyse_path(path)

    assert report.error is not None and report.count == 0


def _face(x=100, y=100, w=200, h=200, eye_distance=70):
    centre_x, eye_y = x + w / 2.0, y + h * 0.38
    return faces.Face(
        box=(x, y, w, h),
        right_eye=(centre_x - eye_distance / 2.0, eye_y),
        left_eye=(centre_x + eye_distance / 2.0, eye_y),
        confidence=0.99,
    )


def _structured(size, seed=0):
    """Hard-edged blobs on a flat base: real structure that blur can remove."""
    rng = np.random.default_rng(seed)
    image = np.full(size, 120, np.uint8)
    for _ in range(24):
        x, y = int(rng.integers(0, max(1, size[1] - 30))), int(rng.integers(0, max(1, size[0] - 30)))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(10, 40)), y + int(rng.integers(10, 40))),
                      int(rng.integers(0, 255)), -1)
    for _ in range(14):
        x, y = int(rng.integers(10, max(11, size[1] - 10))), int(rng.integers(10, max(11, size[0] - 10)))
        cv2.circle(image, (x, y), int(rng.integers(4, 18)), int(rng.integers(0, 255)), -1)
    return image


def test_a_sharp_face_scores_high_a_soft_one_low():
    sharp = np.full((500, 500), 120, np.uint8)
    sharp[100:300, 100:300] = _structured((200, 200))
    soft = np.full((500, 500), 120, np.uint8)
    soft[100:300, 100:300] = cv2.GaussianBlur(_structured((200, 200)), (0, 0), 6.0)

    assert faces.face_sharpness(sharp, _face()) > 60.0 > faces.face_sharpness(soft, _face())


def test_face_sharpness_ignores_the_background():
    """The module's point: a soft face on a razor-sharp background stays soft."""
    soft_face = cv2.GaussianBlur(_structured((200, 200)), (0, 0), 6.0)
    busy = _structured((500, 500), seed=1)
    busy[100:300, 100:300] = soft_face
    plain = np.full((500, 500), 120, np.uint8)
    plain[100:300, 100:300] = soft_face

    assert faces.face_sharpness(busy, _face()) == pytest.approx(faces.face_sharpness(plain, _face()))


def test_tiny_faces_are_skipped_and_boxes_are_clipped():
    image = _structured((300, 300))
    tiny = _face(x=10, y=10, w=config.FACE_MIN_SIZE - 5, h=config.FACE_MIN_SIZE - 5)

    assert faces.face_sharpness(image, tiny) is None
    assert faces.face_sharpness(image, _face(x=200, y=200, w=200, h=200)) is not None


def test_the_main_face_is_the_largest():
    small, large = _face(x=0, y=0, w=60, h=60), _face(x=100, y=100, w=250, h=250)
    large.sharpness, small.sharpness = 88.0, 12.0

    assert faces.FaceReport(faces=[small, large]).main is large
    assert faces.FaceReport(faces=[large, small]).main.sharpness == 88.0
    assert faces.FaceReport().main is None


class _Subject:
    def __init__(self, score, report=None):
        self.score, self.faces = score, report


def test_ranking_uses_the_face_when_there_is_one():
    assert series.rank_score(_Subject(72.0)) == pytest.approx(72.0)
    assert series.rank_score(_Subject(72.0, faces.FaceReport())) == pytest.approx(72.0)

    face = _face()
    face.sharpness = 30.0
    expected = (1 - config.RANK_FACE_WEIGHT) * 90.0 + config.RANK_FACE_WEIGHT * 30.0
    assert series.rank_score(_Subject(90.0, faces.FaceReport(faces=[face]))) == pytest.approx(expected)


def test_the_series_picks_the_sharper_FACE():
    """Background-focused frame scores higher overall, the face-focused one wins."""

    class Member:
        def __init__(self, name, score, face_score, captured):
            self.path, self.score, self.phash, self.captured, self.override = Path(name), score, 0b1010, captured, None
            face = _face()
            face.sharpness = face_score
            self.faces = faces.FaceReport(faces=[face])

    found = series.group_results([
        Member("background_focus.jpg", 96.0, 40.0, 1000.0),
        Member("face_focus.jpg", 90.0, 90.0, 1001.0),
    ])

    assert found[0].best.path.name == "face_focus.jpg"


# ---------------------------------------------------------------------------
# head pose (geometric, so synthetic landmarks check it exactly)
# ---------------------------------------------------------------------------


def _posed(angle=0.0, nose_shift=0.0, eye_distance=70.0):
    rad = np.radians(angle)
    centre = np.array([100.0, 100.0])
    axis = np.array([np.cos(rad), -np.sin(rad)])
    down = np.array([np.sin(rad), np.cos(rad)])
    half = axis * (eye_distance / 2.0)
    return faces.Face(
        box=(50, 50, 100, 100), right_eye=tuple(centre - half), left_eye=tuple(centre + half),
        nose=tuple(centre + axis * nose_shift + down * 20.0),
        mouth_right=(0.0, 0.0), mouth_left=(0.0, 0.0), confidence=0.9,
    )


def test_roll_is_exact_in_degrees():
    for degrees in (-30.0, 0.0, 15.0, 45.0):
        assert faces.head_pose(_posed(angle=degrees))[1] == pytest.approx(degrees, abs=0.01)


def test_yaw_is_proportional_and_independent_of_tilt_and_size():
    assert faces.head_pose(_posed())[0] == pytest.approx(0.0, abs=1e-9)
    # 35 px shift at 70 px eye distance is exactly 0.5
    assert faces.head_pose(_posed(nose_shift=35.0))[0] == pytest.approx(0.5)
    assert faces.head_pose(_posed(angle=35.0, nose_shift=20.0))[0] == pytest.approx(
        faces.head_pose(_posed(nose_shift=20.0))[0], abs=1e-9
    )
    assert faces.head_pose(_posed(nose_shift=10.0, eye_distance=40.0))[0] == pytest.approx(
        faces.head_pose(_posed(nose_shift=25.0, eye_distance=100.0))[0]
    )


def test_facing_the_camera_and_unknown_pose():
    assert _posed(nose_shift=0.0).faces_camera is True
    assert _posed(nose_shift=35.0).faces_camera is False
    no_nose = faces.Face(box=(50, 50, 100, 100), right_eye=(65.0, 90.0), left_eye=(135.0, 90.0), confidence=0.9)
    collapsed = faces.Face(box=(50, 50, 100, 100), right_eye=(100.0, 100.0), left_eye=(100.5, 100.0),
                           nose=(100.0, 120.0), confidence=0.9)

    assert faces.head_pose(no_nose) == (None, None) and no_nose.faces_camera is None
    assert faces.head_pose(collapsed) == (None, None)


def test_facing_the_camera_is_a_ranking_bonus_only():
    def result(face):
        r = sharpness.SharpnessResult(path=None, score=70.0, laplacian=70.0, tenengrad=70.0,
                                      brenner=70.0, low_resolution=False, error=None)
        r.faces = faces.FaceReport(faces=[face])
        return r

    frontal, turned = _posed(nose_shift=0.0), _posed(nose_shift=35.0)
    frontal.sharpness = turned.sharpness = 70.0

    difference = series.rank_score(result(frontal)) - series.rank_score(result(turned))
    assert difference == pytest.approx(config.RANK_FRONTAL_BONUS)


# ---------------------------------------------------------------------------
# one detector per thread (a shared one corrupted 2/3 of the results)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not faces.model_available(), reason="no model file")
def test_every_thread_gets_its_own_detector_and_reuses_it():
    found = []

    def one_thread():
        found.append(faces._detector_for(320, 240))

    # sequentially, so a shared global would deterministically repeat itself
    for _ in range(4):
        worker = threading.Thread(target=one_thread)
        worker.start()
        worker.join()

    assert len({id(d) for d in found}) == 4 and all(d is not None for d in found)
    first = faces._detector_for(320, 240)
    assert faces._detector_for(320, 240) is first and faces._detector_for(640, 480) is first


@pytest.mark.skipif(not faces.model_available(), reason="no model file")
def test_a_size_change_in_another_thread_does_not_disturb_ours():
    ours = faces._detector_for(1024, 683)
    other_size = []

    def other():
        faces._detector_for(683, 1024)
        other_size.append(faces._thread_local.size)

    worker = threading.Thread(target=other)
    worker.start()
    worker.join()

    assert other_size == [(683, 1024)]
    assert faces._thread_local.size == (1024, 683)
    assert faces._detector_for(1024, 683) is ours
