import numpy as np
import pytest
from scipy.signal import sosfreqz

from timbrepro.effects import Delay, Overdrive, Reverb, ThreeBandEQ, _Allpass
from timbrepro.params import BLOCK_SIZE, SAMPLE_RATE as SR


def in_blocks(effect, x, *params):
    return np.concatenate([effect.process(x[i:i + BLOCK_SIZE], *params)
                           for i in range(0, len(x), BLOCK_SIZE)])


def impulse(seconds):
    x = np.zeros(int(seconds * SR))
    x[0] = 1
    return x


def sine(freq, seconds, amplitude=0.2):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)


def gain_db(sos, freq):
    _, h = sosfreqz(sos, worN=[freq], fs=SR)
    return 20 * np.log10(np.abs(h[0]))


def test_eq_is_flat_with_bands_centered():
    sos = ThreeBandEQ().coefficients(0.5, 0.5, 0.5)
    for freq in (50, 200, 800, 3200, 15_000):
        assert gain_db(sos, freq) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("value, expected_db", [(1.0, 12), (0.0, -12), (0.75, 6)])
def test_each_eq_band_reaches_its_gain(value, expected_db):
    eq = ThreeBandEQ()
    assert gain_db(eq.coefficients(value, 0.5, 0.5), 10) == pytest.approx(expected_db, abs=0.05)
    assert gain_db(eq.coefficients(0.5, value, 0.5), 800) == pytest.approx(expected_db, abs=1e-9)
    assert gain_db(eq.coefficients(0.5, 0.5, value), 23_000) == pytest.approx(expected_db, abs=0.05)


def test_overdrive_adds_harmonics_as_gain_rises():
    def third_harmonic_ratio(gain):
        y = in_blocks(Overdrive(), sine(220, 1.0), gain, 1.0, 1.0)[SR // 2:]
        spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        return spectrum[660] / spectrum[220]

    assert third_harmonic_ratio(1.0) > 50 * third_harmonic_ratio(0.0)


def test_overdrive_removes_the_dc_from_asymmetric_clipping():
    y = in_blocks(Overdrive(), sine(220, 2.0), 1.0, 1.0, 1.0)[SR:]
    assert abs(y.mean()) < 1e-3


def test_delay_echo_lands_at_the_programmed_time():
    time = 0.5
    expected_s = Delay.TIME_S[0] * (Delay.TIME_S[1] / Delay.TIME_S[0]) ** time
    y = in_blocks(Delay(), impulse(0.5), time, 0.0, 1.0)
    assert np.argmax(np.abs(y)) == pytest.approx(expected_s * SR, abs=1)


def test_delay_feedback_repeats_and_decays():
    y = in_blocks(Delay(), impulse(0.5), 0.0, 1.0, 1.0)
    period = round(Delay.TIME_S[0] * SR)
    echoes = [np.abs(y[k * period - 2:k * period + 3]).max() for k in (1, 2, 3)]
    assert echoes[0] > echoes[1] > echoes[2] > 0


def test_reverb_tail_rings_and_dies_out():
    y = in_blocks(Reverb(), impulse(8.0), 1.0, 1.0)
    first, last = np.sum(y[:SR] ** 2), np.sum(y[-SR:] ** 2)
    assert first > 0
    assert 0 < last < first / 100


def test_allpass_preserves_energy():
    y = in_blocks(_Allpass(257, 0.5), impulse(4.0))
    assert np.sum(y ** 2) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("effect, params", [(Delay, (0.5, 0.5)), (Reverb, (0.5,))])
def test_zero_mix_is_fully_dry(effect, params):
    x = sine(440, 0.5)
    assert in_blocks(effect(), x, *params, 0.0) == pytest.approx(x)
