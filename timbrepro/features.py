"""Espectrograma mel: a "imagem" do timbre que a CNN recebe.

Feito em numpy para gerar o dataset sem PyTorch nem librosa. Um trecho de 3 s a 48 kHz
vira uma matriz de 128 bandas por 282 quadros.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window

from timbrepro.params import SAMPLE_RATE

N_FFT = 2048
HOP = 512  # 10,7 ms: resolve bem os ecos do delay (20 a 800 ms)
N_MELS = 128
FMAX = 16_000
TOP_DB = 80.0  # faixa dinâmica máxima; abaixo disso é ruído que só atrapalha


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX):
    """Filtros triangulares na escala mel, com área normalizada (Slaney)."""
    fft_hz = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(0, _hz_to_mel(fmax), n_mels + 2))
    low, center, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (fft_hz - low) / (center - low)
    falling = (high - fft_hz) / (high - center)
    return np.maximum(0, np.minimum(rising, falling)) * (2 / (high - low))


_FILTERBANK = mel_filterbank()
_WINDOW = get_window("hann", N_FFT)


def log_mel(audio, sr=SAMPLE_RATE):
    """Espectrograma mel em dB, forma (N_MELS, 1 + len(audio) // HOP), em float32."""
    filterbank = _FILTERBANK if sr == SAMPLE_RATE else mel_filterbank(sr)
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    frames = sliding_window_view(padded, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    db = 10 * np.log10(np.maximum(filterbank @ power.T, 1e-10))
    return np.maximum(db, db.max() - TOP_DB).astype(np.float32)
