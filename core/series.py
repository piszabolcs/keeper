# Keeper - series (burst) detection and picking the best of a series.
#
# Photographers shoot three to five frames per moment. One threshold for the
# whole folder either keeps all of them (a good series) or throws the whole
# moment away (a weak one). Both are solved by recognising which frames belong
# together and which of them is best.

from dataclasses import dataclass

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------
# perceptual hash
# ---------------------------------------------------------------------------

PHASH_DCT_SIZE = 32   # DCT input size
PHASH_BLOCK = 8       # low-frequency corner kept: 8x8 = 64 bits
PHASH_BITS = PHASH_BLOCK * PHASH_BLOCK


def phash(gray):
    """
    64-bit perceptual hash of a grey image: similar-looking frames get similar
    hashes even though their bytes differ completely. The lowest DCT
    frequencies (coarse structure) are compared to their median; a global
    brightness change shifts all of them together and leaves the hash alone.
    """
    small = cv2.resize(
        gray, (PHASH_DCT_SIZE, PHASH_DCT_SIZE), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    coefficients = cv2.dct(small)[:PHASH_BLOCK, :PHASH_BLOCK].flatten()

    # the DC term (mean brightness) dwarfs the rest; leave it out of the median
    middle = float(np.median(coefficients[1:]))

    value = 0
    for bit in coefficients > middle:
        value = (value << 1) | int(bit)
    return value


def hamming(a, b):
    """Differing bits: 0 = looks identical, ~32 = unrelated. Unknown = farthest."""
    if a is None or b is None:
        return PHASH_BITS
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


@dataclass
class Series:
    """A detected series: members in shooting order, and the best one."""

    index: int      # 1-based, shown in the UI
    members: list   # SharpnessResult list
    best: object = None

    @property
    def size(self):
        return len(self.members)


@dataclass
class SeriesInfo:
    """Where one shot sits in its series (what the card displays)."""

    index: int      # which series
    position: int   # 1-based position inside the series
    size: int       # members in the series
    is_best: bool


def rank_score(result):
    """
    Ranking INSIDE a series - a different question from the threshold
    ("better than its neighbours?" instead of "good enough?"), where a wrong
    pick costs little. That is why face sharpness and head pose may count
    here and only here. An unreadable shot gets -1 and is never the best.
    """
    if result.score is None:
        return -1.0

    score = float(result.score)
    report = getattr(result, "faces", None)
    if report is None or not report.faces:
        return score

    main = report.main
    # a portrait is good when the FACE is sharp, not the background
    if main.sharpness is not None:
        weight = config.RANK_FACE_WEIGHT
        score = (1.0 - weight) * score + weight * main.sharpness
    # facing the camera breaks ties but never beats sharpness
    if main.faces_camera:
        score += config.RANK_FRONTAL_BONUS
    return score


def group_results(results):
    """
    Finds series. Two shots belong together when BOTH hold:
      1. taken in quick succession (SERIES_MAX_GAP_SECONDS, EXIF time)
      2. they look alike (SERIES_MAX_DISTANCE)
    Time alone would glue a busy event together; similarity alone would join
    two tripod shots taken hours apart. Shots without an EXIF capture time
    are never grouped: file times change when files are copied (C-13).

    Similarity is measured against the FIRST member (no drifting along a slow
    pan); the time gap against the PREVIOUS member (a long burst stays whole).
    """
    usable = [r for r in results if r.phash is not None and r.captured is not None]
    # same second (EXIF stores seconds): the file name - a camera counter - decides
    ordered = sorted(usable, key=lambda r: (r.captured, r.path.name))

    groups, current = [], []
    for result in ordered:
        if current:
            gap = abs(result.captured - current[-1].captured)
            distance = hamming(current[0].phash, result.phash)
            if gap <= config.SERIES_MAX_GAP_SECONDS and distance <= config.SERIES_MAX_DISTANCE:
                current.append(result)
                continue
            groups.append(current)
        current = [result]
    if current:
        groups.append(current)

    found = []
    for members in groups:
        if len(members) >= config.SERIES_MIN_SIZE:
            found.append(Series(index=len(found) + 1, members=members, best=max(members, key=rank_score)))
    return found


def info_map(found):
    """path -> SeriesInfo, so a card finds its place with one lookup."""
    mapping = {}
    for series in found:
        for position, member in enumerate(series.members, start=1):
            mapping[member.path] = SeriesInfo(
                index=series.index, position=position, size=series.size,
                is_best=member is series.best,
            )
    return mapping


# ---------------------------------------------------------------------------
# bulk actions - they set the same flags a click on a card would set
# ---------------------------------------------------------------------------


def protect_best(found):
    """Keeps the best of every series regardless of the threshold."""
    count = 0
    for series in found:
        if series.best is not None:
            series.best.override = True
            count += 1
    return count


def keep_best_n(found, n=1):
    """
    Keeps the best N of every series and marks the rest as DUPLICATES (they go
    to duplicates/, not rejected/: they are fine frames, only surplus).
    Idempotent: running again with another N replaces the marks.
    Returns (kept, marked).
    """
    kept = marked = 0
    for series in found:
        readable = [m for m in series.members if m.score is not None]
        for place, member in enumerate(sorted(readable, key=rank_score, reverse=True)):
            member.duplicate = place >= max(1, n)
            if member.duplicate:
                marked += 1
            else:
                kept += 1
    return kept, marked


def clear_duplicates(results):
    """Removes every duplicate mark; returns how many shots it touched."""
    count = 0
    for result in results:
        if result.duplicate:
            result.duplicate = False
            count += 1
    return count
