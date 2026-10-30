"""A cadeia de efeitos: a mesma classe serve ao tempo real e à geração do dataset."""

import numpy as np

from timbrepro.effects import Delay, Overdrive, Reverb, ThreeBandEQ
from timbrepro.params import BLOCK_SIZE, SAMPLE_RATE

SMOOTHING_S = 0.05  # abaixo disso, a troca dos coeficientes dos filtros começa a estalar


class FxChain:
    """Overdrive → EQ → delay → reverb, com troca de parâmetros sem clique."""

    def __init__(self, params, sr=SAMPLE_RATE):
        self._target = np.asarray(params, dtype=float)
        self._current = self._target.copy()
        self._smoothing_samples = SMOOTHING_S * sr
        self._overdrive = Overdrive(sr)
        self._eq = ThreeBandEQ(sr)
        self._delay = Delay(sr)
        self._reverb = Reverb(sr)

    def set_params(self, params):
        """Troca o preset; pode ser chamado de outra thread com o áudio rodando."""
        self._target = np.asarray(params, dtype=float)

    def process(self, x):
        """Processa um trecho de qualquer tamanho e devolve a saída limitada a ±1."""
        out = np.empty(len(x))
        for start in range(0, len(x), BLOCK_SIZE):
            y = x[start:start + BLOCK_SIZE]
            step = 1 - np.exp(-len(y) / self._smoothing_samples)
            self._current = self._current + step * (self._target - self._current)
            p = self._current
            y = self._overdrive.process(y, *p[0:3])
            y = self._eq.process(y, *p[3:6])
            y = self._delay.process(y, *p[6:9])
            out[start:start + BLOCK_SIZE] = self._reverb.process(y, *p[9:11])
        return np.clip(out, -1, 1)
