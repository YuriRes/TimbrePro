"""Sinal de teste: guitarra limpa sintética, enquanto não há gravações DI."""

import numpy as np
from scipy.signal import lfilter

from timbrepro.params import SAMPLE_RATE

RIFF_HZ = (164.81, 196.00, 220.00, 246.94, 293.66, 246.94, 220.00, 196.00)


def pluck(freq, seconds, sr=SAMPLE_RATE, seed=0):
    """Corda dedilhada por Karplus-Strong."""
    period = int(sr / freq)
    excitation = np.zeros(int(seconds * sr))
    excitation[:period] = np.random.default_rng(seed).uniform(-1, 1, period)
    a = np.zeros(period + 2)
    a[0] = 1
    a[period] = a[period + 1] = -0.4975
    return lfilter([1.0], a, excitation)


def synth_di(note_seconds=0.4, sr=SAMPLE_RATE):
    """Um riff curto de notas sobrepostas, com pico em 0,5 como um DI bem regulado."""
    note = int(note_seconds * sr)
    out = np.zeros(note * len(RIFF_HZ) + sr)
    for i, freq in enumerate(RIFF_HZ):
        out[i * note:i * note + sr] += pluck(freq, 1.0, sr, seed=i)
    return out / np.abs(out).max() * 0.5
