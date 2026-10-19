# Keeper - the optional learned denoiser. Most tests check that its absence
# is handled well (they run everywhere); the real filtering runs only where
# PyTorch, a CUDA GPU and the weights are present.

import builtins
from pathlib import Path

import numpy as np
import pytest

import config
from core import enhance, learned_denoise


def test_the_reason_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(learned_denoise, "weights_path", lambda: Path("missing_weights.pth"))

    reason = learned_denoise.unavailable_reason(refresh=True)

    assert reason is not None
    assert "weights" in reason or "PyTorch" in reason or "CUDA" in reason
    assert not learned_denoise.available()


def test_without_torch_it_says_so(monkeypatch):
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    try:
        assert "PyTorch" in learned_denoise.unavailable_reason(refresh=True)
    finally:
        monkeypatch.undo()
        learned_denoise.unavailable_reason(refresh=True)


def _noisy(sigma, seed):
    rng = np.random.default_rng(seed)
    return np.clip(np.full((120, 160, 3), 110, np.float32) + rng.normal(0, sigma, (120, 160, 3)), 0, 255).astype(np.uint8)


def test_the_chain_falls_back_to_the_classic_filter(monkeypatch):
    monkeypatch.setattr(learned_denoise, "available", lambda: False)
    image = _noisy(14, 0)

    result = enhance.enhance_image(image, 14.0, 80.0, learned=True)

    assert result.learned is False and result.image.shape == image.shape


def test_the_network_runs_only_when_switched_on_and_needed(monkeypatch):
    calls = []
    monkeypatch.setattr(learned_denoise, "available", lambda: True)
    monkeypatch.setattr(learned_denoise, "denoise", lambda image, cancel=None: calls.append(image) or image)

    # a clean frame: nothing to filter, a minute per photo would be wasted
    assert enhance.enhance_image(np.full((120, 160, 3), 128, np.uint8), 0.0, 80.0, learned=True).learned is False
    # switched off
    assert enhance.enhance_image(_noisy(18, 1), 18.0, 80.0, learned=False).learned is False
    assert calls == []
    # switched on and needed
    assert enhance.enhance_image(_noisy(18, 1), 18.0, 80.0, learned=True).learned is True
    assert len(calls) == 1


available = pytest.mark.skipif(
    not learned_denoise.available(), reason="needs PyTorch, an NVIDIA GPU and the weights file"
)


@available
def test_filtering_keeps_shape_and_brightness():
    image = np.clip(np.random.default_rng(2).normal(120, 20, (300, 400, 3)), 0, 255).astype(np.uint8)
    out = learned_denoise.denoise(image)

    assert out.shape == image.shape and out.dtype == np.uint8
    assert float(out.std()) < float(image.std())
    assert float(out.mean()) == pytest.approx(float(image.mean()), abs=4.0)


@available
def test_no_seams_at_tile_borders():
    size = config.LEARNED_TILE + 200
    image = np.clip(np.random.default_rng(3).normal(120, 22, (size, size, 3)), 0, 255).astype(np.uint8)
    out = learned_denoise.denoise(image).astype(np.float32).mean(axis=2)

    border = config.LEARNED_TILE
    jumps = np.abs(np.diff(out[:, border - 4:border + 4], axis=1)).mean(axis=0)
    assert jumps[3] < 2.0 * float(np.median(jumps))


@available
def test_a_small_image_fits_one_tile():
    small = np.full((64, 64, 3), 100, np.uint8)
    assert learned_denoise.denoise(small).shape == small.shape
