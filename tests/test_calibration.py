# Keeper - calibration regression test.
#
# Twice the program ran green with a useless score: with the phase-1
# constants all 68 real frames scored exactly 100, and with raw (not
# contrast-normalised) metrics they were crammed into 78-99 in REVERSED order.
# These tests run on measured metric values of real frames (no image files),
# so the sample photos never enter the repository.

import config
from core.sharpness import expectation_offset, normalize_log

# contrast-normalised, noise-corrected metrics of 68 real frames at 1024 px
REAL_MEDIAN = {"laplacian": 0.234005, "tenengrad": 2.715518, "brenner": 0.101278}
REAL_SHARPEST = {"laplacian": 0.437337, "tenengrad": 4.213405, "brenner": 0.150576}

# the same 68 frames with growing Gaussian blur (median per metric)
BLUR_SERIES = [
    (0.0, {"laplacian": 0.214582, "tenengrad": 2.846566, "brenner": 0.107137}),
    (0.5, {"laplacian": 0.096883, "tenengrad": 2.286971, "brenner": 0.080233}),
    (1.0, {"laplacian": 0.015105, "tenengrad": 1.299968, "brenner": 0.041794}),
    (1.5, {"laplacian": 0.004514, "tenengrad": 0.827641, "brenner": 0.025933}),
    (2.0, {"laplacian": 0.001796, "tenengrad": 0.569758, "brenner": 0.017811}),
]

# groups checked BY EYE at 1:1 (soft: through a windscreen, shaken shoe;
# sharp: number plate letters, paving texture)
VISUALLY_SOFT = {
    "DIK09984": {"laplacian": 0.11489, "tenengrad": 1.604201, "brenner": 0.063021},
    "DIK09863": {"laplacian": 0.148196, "tenengrad": 1.428846, "brenner": 0.068883},
    "DIK09978": {"laplacian": 0.157263, "tenengrad": 1.734112, "brenner": 0.071468},
    "DIK09938": {"laplacian": 0.161883, "tenengrad": 2.688442, "brenner": 0.099589},
    "DIK09971": {"laplacian": 0.164466, "tenengrad": 2.726403, "brenner": 0.100758},
    "DIK09965": {"laplacian": 0.174862, "tenengrad": 2.730971, "brenner": 0.101582},
}
VISUALLY_SHARP = {
    "DIK09850": {"laplacian": 0.258402, "tenengrad": 3.346518, "brenner": 0.128536},
    "DIK09886": {"laplacian": 0.265868, "tenengrad": 3.133824, "brenner": 0.117077},
    "DIK09838": {"laplacian": 0.294852, "tenengrad": 3.057011, "brenner": 0.117276},
    "DIK09857": {"laplacian": 0.295923, "tenengrad": 3.559108, "brenner": 0.138393},
    "DIK09992": {"laplacian": 0.418254, "tenengrad": 3.878814, "brenner": 0.145501},
    "DIK09845": {"laplacian": 0.437337, "tenengrad": 4.213405, "brenner": 0.150576},
}

RANGES = {
    "laplacian": (config.LAPLACIAN_LO, config.LAPLACIAN_HI),
    "tenengrad": (config.TENENGRAD_LO, config.TENENGRAD_HI),
    "brenner": (config.BRENNER_LO, config.BRENNER_HI),
}
WEIGHTS = {
    "laplacian": config.WEIGHT_LAPLACIAN,
    "tenengrad": config.WEIGHT_TENENGRAD,
    "brenner": config.WEIGHT_BRENNER,
}


def parts(values):
    return {name: normalize_log(v, *RANGES[name]) for name, v in values.items()}


def total(values):
    return sum(score * WEIGHTS[name] for name, score in parts(values).items())


def gap():
    return min(total(v) for v in VISUALLY_SHARP.values()) - max(
        total(v) for v in VISUALLY_SOFT.values()
    )


def test_no_metric_sticks_to_100_on_the_sharpest_frame():
    for name, score in parts(REAL_SHARPEST).items():
        assert score < 100.0, "%s sticks to 100, HI is too low" % name


def test_no_metric_sticks_to_0_on_the_median_frame():
    for name, score in parts(REAL_MEDIAN).items():
        assert score > 0.0, "%s sticks to 0, LO is too high" % name


def test_the_three_metrics_agree_on_one_frame():
    scores = parts(REAL_MEDIAN)
    assert max(scores.values()) - min(scores.values()) <= 15.0


def test_visually_soft_frames_score_below_visually_sharp_ones_with_a_real_gap():
    # the most important test: the number agrees with the eye
    assert gap() >= 15.0


def test_the_shooting_conditions_can_never_flip_a_frame():
    """The largest EXIF shift must stay below the soft/sharp gap."""
    assert config.STRICT_PENALTY + config.LENIENT_BONUS < gap()


def test_the_default_threshold_separates_the_groups():
    # the reference set is all sunny ISO 100, i.e. the strict side
    for name, values in VISUALLY_SOFT.items():
        assert total(values) - config.STRICT_PENALTY < config.DEFAULT_THRESHOLD, name
    for name, values in VISUALLY_SHARP.items():
        assert total(values) - config.STRICT_PENALTY >= config.DEFAULT_THRESHOLD, name


def test_the_score_falls_monotonically_with_blur():
    scores = [total(v) for _, v in BLUR_SERIES]
    assert all(b <= a for a, b in zip(scores, scores[1:])), scores


def test_the_blur_series_uses_the_whole_scale():
    scores = [total(v) for _, v in BLUR_SERIES]
    assert max(scores) - min(scores) >= 50.0
    assert scores[4] <= 5.0                            # sigma 2.0: bottom of the scale
    assert scores[2] < config.DEFAULT_THRESHOLD        # sigma 1.0: visibly soft
    assert scores[1] > 5.0                             # sigma 0.5: still usable


class _Shot:
    """Minimal ShotInfo stand-in."""

    def __init__(self, iso=None, exposure=None, focal=None):
        self.iso, self.exposure, self.focal35 = iso, exposure, focal

    @property
    def effective_focal(self):
        return self.focal35


def test_ideal_conditions_are_stricter():
    assert expectation_offset(_Shot(iso=100, exposure=1 / 500.0, focal=50)) == config.STRICT_PENALTY


def test_high_iso_or_slow_shutter_is_more_lenient_but_not_twice():
    only_iso = expectation_offset(_Shot(iso=6400, exposure=1 / 500.0, focal=50))
    only_shutter = expectation_offset(_Shot(iso=100, exposure=1 / 25.0, focal=200))
    both = expectation_offset(_Shot(iso=6400, exposure=1 / 25.0, focal=200))

    assert only_iso == only_shutter == both == -config.LENIENT_BONUS


def test_missing_data_is_neutral():
    # manual lenses do not record a focal length: no reason for strictness
    assert expectation_offset(_Shot(iso=100, exposure=1 / 80.0, focal=None)) == 0.0
    assert expectation_offset(None) == 0.0
    assert expectation_offset(_Shot()) == 0.0
