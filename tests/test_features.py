import numpy as np

from timbrepro.features import N_FFT, N_MELS, TOP_DB, log_mel, mel_filterbank
from timbrepro.params import SAMPLE_RATE as SR


def test_filterbank_shape_and_no_empty_band():
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, N_FFT // 2 + 1)
    assert np.all(fb >= 0)
    assert np.all(fb.sum(axis=1) > 0)


def test_log_mel_shape_for_a_three_second_clip():
    mel = log_mel(np.random.default_rng(0).uniform(-0.5, 0.5, 3 * SR))
    assert mel.shape == (128, 282)
    assert mel.dtype == np.float32


def test_sine_peaks_at_its_mel_band():
    t = np.arange(3 * SR) / SR
    mel = log_mel(0.5 * np.sin(2 * np.pi * 1000 * t))
    peak_hz = mel_filterbank().argmax(axis=1) * SR / N_FFT
    assert mel.mean(axis=1).argmax() == np.abs(peak_hz - 1000).argmin()


def test_silence_is_finite():
    assert np.all(np.isfinite(log_mel(np.zeros(3 * SR))))


def test_dynamic_range_is_capped():
    t = np.arange(3 * SR) / SR
    mel = log_mel(0.5 * np.sin(2 * np.pi * 440 * t))
    assert mel.max() - mel.min() <= TOP_DB + 1e-3
