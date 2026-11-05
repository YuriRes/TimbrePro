"""Geração do dataset: DI + parâmetros sorteados → som, pela mesma FxChain do tempo real.

A CNN aprende o caminho inverso (som → parâmetros), então cada exemplo é um trecho de DI
processado com um vetor de parâmetros conhecido. Não há rotulagem manual.
"""

import numpy as np

from timbrepro.chain import FxChain
from timbrepro.params import PARAM_NAMES, SAMPLE_RATE

CLIP_S = 3  # duração do trecho que vira exemplo
PREROLL_S = 1  # áudio anterior ao trecho, para delay e reverb já estarem "cheios"
MIX_OFF_PROB = 0.3  # presets reais muitas vezes não usam delay ou reverb
PEAK = 0.9  # o volume da referência não diz nada sobre o timbre, então normalizamos

_DELAY_MIX = PARAM_NAMES.index("delay_mix")
_REVERB_MIX = PARAM_NAMES.index("reverb_mix")


def sample_params(rng):
    """Sorteia um vetor de parâmetros uniforme em [0, 1], às vezes sem delay ou reverb."""
    params = rng.uniform(0, 1, len(PARAM_NAMES))
    if rng.random() < MIX_OFF_PROB:
        params[_DELAY_MIX] = 0.0
    if rng.random() < MIX_OFF_PROB:
        params[_REVERB_MIX] = 0.0
    return params


def split_clips(audio, sr=SAMPLE_RATE, min_rms_db=-40.0):
    """Corta o DI em janelas de pré-rolagem + trecho, pulando trechos em silêncio."""
    window, hop = (PREROLL_S + CLIP_S) * sr, CLIP_S * sr
    clips = []
    for start in range(0, len(audio) - window + 1, hop):
        clip = audio[start:start + window]
        rms = np.sqrt(np.mean(clip[-hop:] ** 2))
        if 20 * np.log10(max(rms, 1e-12)) >= min_rms_db:
            clips.append(clip)
    return clips


def render_example(dry, params, sr=SAMPLE_RATE):
    """Processa a janela inteira e devolve só o trecho final, com pico normalizado."""
    wet = FxChain(params, sr).process(dry)[-CLIP_S * sr:]
    peak = np.abs(wet).max()
    return wet * (PEAK / peak) if peak >= 1e-9 else np.zeros_like(wet)
