import numpy as np
import pytest

from timbrepro.dataset import CLIP_S, PEAK, PREROLL_S, render_example, sample_params, split_clips
from timbrepro.params import PARAM_NAMES, SAMPLE_RATE as SR

DELAY_MIX = PARAM_NAMES.index("delay_mix")
REVERB_MIX = PARAM_NAMES.index("reverb_mix")
REVERB_SIZE = PARAM_NAMES.index("reverb_size")
WINDOW = (PREROLL_S + CLIP_S) * SR


def noise(seconds, amplitude=0.3, seed=0):
    return amplitude * np.random.default_rng(seed).uniform(-1, 1, int(seconds * SR))


def test_sample_params_shape_and_range():
    rng = np.random.default_rng(0)
    draws = np.array([sample_params(rng) for _ in range(1000)])
    assert draws.shape == (1000, 11)
    assert np.all((draws >= 0) & (draws <= 1))


def test_sample_params_turns_mixes_off_sometimes():
    rng = np.random.default_rng(1)
    draws = np.array([sample_params(rng) for _ in range(2000)])
    assert 0.2 < np.mean(draws[:, DELAY_MIX] == 0) < 0.4
    assert 0.2 < np.mean(draws[:, REVERB_MIX] == 0) < 0.4


def test_sample_params_is_reproducible():
    first = sample_params(np.random.default_rng(42))
    second = sample_params(np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


def test_split_clips_window_length_and_count():
    clips = split_clips(noise(10))
    assert len(clips) == 3
    assert all(len(clip) == WINDOW for clip in clips)


def test_split_clips_skips_silent_windows():
    audio = np.concatenate([noise(4), np.zeros(9 * SR)])
    assert len(split_clips(audio)) == 1


def test_split_clips_short_audio_gives_nothing():
    assert split_clips(noise(2)) == []


def test_render_example_length_and_peak():
    params = sample_params(np.random.default_rng(3))
    wet = render_example(noise(PREROLL_S + CLIP_S), params)
    assert len(wet) == CLIP_S * SR
    assert np.abs(wet).max() == pytest.approx(PEAK, abs=1e-6)


def test_render_example_uses_the_preroll():
    dry = np.zeros(WINDOW)
    dry[0] = 1.0
    params = np.zeros(11)
    params[[PARAM_NAMES.index("drive_level"), REVERB_MIX, REVERB_SIZE]] = 1.0
    params[[PARAM_NAMES.index(n) for n in ("eq_low", "eq_mid", "eq_high")]] = 0.5
    wet = render_example(dry, params)
    assert np.abs(wet).max() > 0


def test_render_example_of_silence_is_zero():
    wet = render_example(np.zeros(WINDOW), sample_params(np.random.default_rng(4)))
    assert np.all(np.isfinite(wet))
    assert np.all(wet == 0)
