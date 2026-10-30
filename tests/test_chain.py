from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from timbrepro.chain import FxChain
from timbrepro.live import StreamStats, make_callback
from timbrepro.params import BLOCK_SIZE, PARAM_NAMES, SAMPLE_RATE as SR, load_preset, params_from_dict

PRESETS = sorted(Path(__file__).parent.parent.glob("presets/*.json"))


def noise(seconds, amplitude=0.3, seed=0):
    return amplitude * np.random.default_rng(seed).uniform(-1, 1, int(seconds * SR))


def test_there_are_eleven_params():
    assert len(PARAM_NAMES) == 11


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_bundled_presets_are_valid(path):
    assert load_preset(path).shape == (11,)


@pytest.mark.parametrize("change", [
    {"reverb_mix": 1.5},
    {"reverb_mix": float("nan")},
])
def test_preset_with_value_out_of_range_is_rejected(change):
    with pytest.raises(ValueError):
        params_from_dict(dict.fromkeys(PARAM_NAMES, 0.5) | change)


def test_preset_with_wrong_names_is_rejected():
    values = dict.fromkeys(PARAM_NAMES, 0.5)
    del values["eq_mid"]
    values["eq_middle"] = 0.5
    with pytest.raises(ValueError, match="eq_mid.*eq_middle"):
        params_from_dict(values)


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.stem)
def test_output_does_not_depend_on_how_audio_is_split_into_blocks(path):
    """Tempo real e dataset cortam o áudio de formas diferentes e precisam soar igual."""
    params = load_preset(path)
    x = noise(1.0)
    whole = FxChain(params).process(x)

    chain = FxChain(params)
    rng = np.random.default_rng(1)
    pieces, start = [], 0
    while start < len(x):
        size = int(rng.integers(1, 2 * BLOCK_SIZE))
        pieces.append(chain.process(x[start:start + size]))
        start += size

    assert np.concatenate(pieces) == pytest.approx(whole, abs=1e-9)


def test_all_params_at_maximum_stay_stable():
    chain = FxChain(np.ones(11))
    during = chain.process(noise(2.0))
    after = chain.process(np.zeros(12 * SR))
    assert np.all(np.isfinite(after))
    assert np.sqrt(np.mean(after[-SR:] ** 2)) < np.sqrt(np.mean(during ** 2)) / 2


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_changing_a_param_during_playback_does_not_click(name):
    """Um clique é energia em alta frequência que uma senoide pura não teria.

    A troca cai no pico do cosseno: num cruzamento por zero, o salto quase não aparece.
    """
    start = dict.fromkeys(PARAM_NAMES, 0.5) | {"drive_gain": 0.1, "delay_mix": 0.5, "reverb_mix": 0.5}
    x = 0.2 * np.cos(2 * np.pi * 220 * np.arange(3 * SR) / SR)

    chain = FxChain(params_from_dict(start | {name: 0.0}))
    before = chain.process(x[:2 * SR])
    chain.set_params(params_from_dict(start | {name: 1.0}))
    after = chain.process(x[2 * SR:])

    high_band = np.abs(sosfilt(butter(4, 6000, "highpass", fs=SR, output="sos"),
                               np.concatenate([before, after])))
    steady_db = 20 * np.log10(max(high_band[SR:2 * SR].max(), high_band[-SR // 4:].max()))
    switch_db = 20 * np.log10(high_band[2 * SR:2 * SR + SR // 4].max())
    assert switch_db < max(steady_db + 3, -50)


def test_callback_fills_both_outputs_and_counts_xruns():
    stats = StreamStats()
    callback = make_callback(lambda x: x * 2, stats)
    indata = np.full((BLOCK_SIZE, 1), 0.25, dtype=np.float32)
    outdata = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)

    callback(indata, outdata, BLOCK_SIZE, None, False)
    assert np.all(outdata == 0.5)
    assert (stats.blocks, stats.xruns) == (1, 0)

    callback(indata, outdata, BLOCK_SIZE, None, True)
    assert (stats.blocks, stats.xruns) == (2, 1)
