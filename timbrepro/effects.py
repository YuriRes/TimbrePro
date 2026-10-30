"""Os 4 efeitos da cadeia: overdrive, EQ de 3 bandas, delay e reverb.

Todos processam áudio mono em float64, um bloco por vez, e guardam o estado entre
blocos. Os parâmetros chegam normalizados de 0 a 1. Blocos têm no máximo BLOCK_SIZE
amostras (FxChain garante isso), o que mantém delay e reverb vetorizados: a 48 kHz, toda
linha de atraso é mais longa que um bloco, então nunca se lê o que acabou de ser escrito.
"""

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from timbrepro.params import SAMPLE_RATE


class _Ramp:
    """Leva um valor até o alvo ao longo do bloco, para a mudança não virar degrau."""

    def __init__(self):
        self._value = None

    def __call__(self, target, n):
        start = target if self._value is None else self._value
        self._value = target
        return start + (target - start) * np.arange(1, n + 1) / n


# Biquads do Audio EQ Cookbook (RBJ). Cada função devolve uma linha de matriz SOS.

def _section(b0, b1, b2, a0, a1, a2):
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def lowpass(freq, q, sr):
    w0 = 2 * np.pi * freq / sr
    cos_w, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    return _section((1 - cos_w) / 2, 1 - cos_w, (1 - cos_w) / 2,
                    1 + alpha, -2 * cos_w, 1 - alpha)


def low_shelf(freq, gain_db, sr):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cos_w, beta = np.cos(w0), np.sqrt(2 * a) * np.sin(w0)
    return _section(a * ((a + 1) - (a - 1) * cos_w + beta),
                    2 * a * ((a - 1) - (a + 1) * cos_w),
                    a * ((a + 1) - (a - 1) * cos_w - beta),
                    (a + 1) + (a - 1) * cos_w + beta,
                    -2 * ((a - 1) + (a + 1) * cos_w),
                    (a + 1) + (a - 1) * cos_w - beta)


def high_shelf(freq, gain_db, sr):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cos_w, beta = np.cos(w0), np.sqrt(2 * a) * np.sin(w0)
    return _section(a * ((a + 1) + (a - 1) * cos_w + beta),
                    -2 * a * ((a - 1) + (a + 1) * cos_w),
                    a * ((a + 1) + (a - 1) * cos_w - beta),
                    (a + 1) - (a - 1) * cos_w + beta,
                    2 * ((a - 1) - (a + 1) * cos_w),
                    (a + 1) - (a - 1) * cos_w - beta)


def peaking(freq, q, gain_db, sr):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    cos_w, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    return _section(1 + alpha * a, -2 * cos_w, 1 - alpha * a,
                    1 + alpha / a, -2 * cos_w, 1 - alpha / a)


class Overdrive:
    """Pré-ganho, waveshaping tanh assimétrico e filtro de tom passa-baixas."""

    MAX_GAIN_DB = 40
    BIAS = 0.2  # a assimetria gera harmônicos pares (e DC, removido em seguida)
    TONE_HZ = (800, 12_000)

    def __init__(self, sr=SAMPLE_RATE):
        self._sr = sr
        self._dc_block = butter(1, 20, "highpass", fs=sr, output="sos")[0].tolist()
        self._zi = np.zeros((2, 2))
        self._gain = _Ramp()
        self._level = _Ramp()

    def process(self, x, gain, tone, level):
        low, high = self.TONE_HZ
        pre = self._gain(10 ** (gain * self.MAX_GAIN_DB / 20), len(x))
        y = np.tanh(x * pre + self.BIAS) - np.tanh(self.BIAS)
        sos = np.array([self._dc_block, lowpass(low * (high / low) ** tone, 0.707, self._sr)])
        y, self._zi = sosfilt(sos, y, zi=self._zi)
        return y * self._level(level, len(x))


class ThreeBandEQ:
    """Low shelf, peaking e high shelf. O valor 0,5 em uma banda significa 0 dB."""

    RANGE_DB = 12
    LOW_HZ, MID_HZ, HIGH_HZ = 200, 800, 3200
    MID_Q = 0.7

    def __init__(self, sr=SAMPLE_RATE):
        self._sr = sr
        self._zi = np.zeros((3, 2))

    def coefficients(self, low, mid, high):
        low_db, mid_db, high_db = (2 * self.RANGE_DB * (v - 0.5) for v in (low, mid, high))
        return np.array([low_shelf(self.LOW_HZ, low_db, self._sr),
                         peaking(self.MID_HZ, self.MID_Q, mid_db, self._sr),
                         high_shelf(self.HIGH_HZ, high_db, self._sr)])

    def process(self, x, low, mid, high):
        y, self._zi = sosfilt(self.coefficients(low, mid, high), x, zi=self._zi)
        return y


class Delay:
    """Buffer circular com realimentação filtrada e leitura interpolada."""

    TIME_S = (0.02, 0.8)
    MAX_FEEDBACK = 0.9
    FEEDBACK_LOWPASS_HZ = 3500

    def __init__(self, sr=SAMPLE_RATE):
        self._sr = sr
        self._buffer = np.zeros(int(self.TIME_S[1] * sr) + 2)
        self._write = 0
        self._b, self._a = butter(1, self.FEEDBACK_LOWPASS_HZ, fs=sr)
        self._zi = np.zeros(1)
        self._delay = _Ramp()
        self._mix = _Ramp()

    def process(self, x, time, feedback, mix):
        n, size = len(x), len(self._buffer)
        low, high = self.TIME_S
        # Mover o ponto de leitura aos poucos troca o clique por uma varredura de altura.
        delay = self._delay(low * (high / low) ** time * self._sr, n)
        write = self._write + np.arange(n)
        read = write - delay
        floor = np.floor(read).astype(int)
        frac = read - floor
        wet = (1 - frac) * self._buffer[floor % size] + frac * self._buffer[(floor + 1) % size]

        filtered, self._zi = lfilter(self._b, self._a, wet, zi=self._zi)
        self._buffer[write % size] = x + feedback * self.MAX_FEEDBACK * filtered
        self._write = (self._write + n) % size

        wet_gain = self._mix(mix, n)
        return (1 - wet_gain) * x + wet_gain * wet


class _Comb:
    """Filtro pente com passa-baixas na realimentação (Freeverb)."""

    def __init__(self, length, damping):
        self._buffer = np.zeros(length)
        self._position = 0
        self._b, self._a = [1 - damping], [1, -damping]
        self._zi = np.zeros(1)

    def process(self, x, feedback):
        index = (self._position + np.arange(len(x))) % len(self._buffer)
        delayed = self._buffer[index]
        damped, self._zi = lfilter(self._b, self._a, delayed, zi=self._zi)
        self._buffer[index] = x + feedback * damped
        self._position = (index[-1] + 1) % len(self._buffer)
        return delayed


class _Allpass:
    """Passa-tudo de Schroeder: espalha os ecos sem mudar a magnitude."""

    def __init__(self, length, gain):
        self._buffer = np.zeros(length)
        self._position = 0
        self._gain = gain

    def process(self, x):
        index = (self._position + np.arange(len(x))) % len(self._buffer)
        delayed = self._buffer[index]
        v = x + self._gain * delayed
        self._buffer[index] = v
        self._position = (index[-1] + 1) % len(self._buffer)
        return delayed - self._gain * v


class Reverb:
    """Freeverb mono: 8 pentes em paralelo seguidos de 4 passa-tudo em série."""

    # Afinações do Freeverb, em amostras a 44,1 kHz. O último passa-tudo era 225; com 236
    # ele tem 257 amostras a 48 kHz e continua maior que BLOCK_SIZE.
    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASSES = (556, 441, 341, 236)
    DAMPING = 0.2
    ALLPASS_GAIN = 0.5

    def __init__(self, sr=SAMPLE_RATE):
        scale = sr / 44_100
        self._combs = [_Comb(round(n * scale), self.DAMPING) for n in self.COMBS]
        self._allpasses = [_Allpass(round(n * scale), self.ALLPASS_GAIN) for n in self.ALLPASSES]
        self._mix = _Ramp()

    def process(self, x, size, mix):
        feedback = 0.7 + 0.28 * size
        wet = sum(comb.process(x, feedback) for comb in self._combs)
        # Mantém a energia da cauda perto da do sinal seco, para qualquer tamanho.
        wet = wet * np.sqrt((1 - feedback ** 2) / len(self._combs))
        for allpass in self._allpasses:
            wet = allpass.process(wet)
        wet_gain = self._mix(mix, len(x))
        return (1 - wet_gain) * x + wet_gain * wet
