"""Os 11 parâmetros da cadeia e a leitura de presets JSON.

O vetor de parâmetros é a interface entre o motor DSP e a IA: a CNN vai prever
exatamente este vetor, na ordem de PARAM_NAMES, com valores normalizados de 0 a 1.
"""

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
BLOCK_SIZE = 256  # 5,33 ms a 48 kHz; a cadeia corta trechos maiores neste tamanho

PARAM_NAMES = (
    "drive_gain", "drive_tone", "drive_level",
    "eq_low", "eq_mid", "eq_high",
    "delay_time", "delay_feedback", "delay_mix",
    "reverb_size", "reverb_mix",
)


def params_from_dict(values):
    """Converte {nome: valor} no vetor de parâmetros, validando nomes e faixa."""
    missing = sorted(set(PARAM_NAMES) - set(values))
    unknown = sorted(set(values) - set(PARAM_NAMES))
    if missing or unknown:
        raise ValueError(f"preset inválido: faltando {missing}, desconhecidos {unknown}")
    params = np.array([values[name] for name in PARAM_NAMES], dtype=float)
    if not np.all((params >= 0) & (params <= 1)):
        raise ValueError("preset inválido: os parâmetros devem estar entre 0 e 1")
    return params


def load_preset(path):
    """Lê um preset JSON no formato {"name": ..., "params": {nome: valor}}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return params_from_dict(data["params"])
