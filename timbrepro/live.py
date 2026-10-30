"""Stream em tempo real: guitarra → cadeia de efeitos → saída."""

import time
from dataclasses import dataclass

import numpy as np

from timbrepro.chain import FxChain
from timbrepro.params import SAMPLE_RATE, load_preset


@dataclass
class StreamStats:
    blocks: int = 0
    xruns: int = 0
    worst_s: float = 0.0


def make_callback(process, stats):
    """Monta o callback de áudio em torno de `process`, um bloco mono por chamada."""

    def callback(indata, outdata, frames, time_info, status):
        start = time.perf_counter()
        if status:
            stats.xruns += 1
        outdata[:] = process(indata[:, 0].astype(np.float64))[:, np.newaxis]
        stats.blocks += 1
        stats.worst_s = max(stats.worst_s, time.perf_counter() - start)

    return callback


def run(preset, block_size, device=None, bypass=False):
    """Abre o stream e aceita trocas de preset pelo terminal até receber Enter vazio."""
    # Importado aqui para o resto do pacote funcionar sem PortAudio.
    import sounddevice as sd

    chain = FxChain(load_preset(preset))
    stats = StreamStats()
    process = (lambda x: x) if bypass else chain.process

    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=block_size, device=device,
                   channels=(1, 2), dtype="float32", latency="low",
                   callback=make_callback(process, stats)) as stream:
        latency_ms = sum(stream.latency) * 1000
        print(f"Rodando: bloco de {block_size} amostras, latência reportada {latency_ms:.1f} ms")
        print("Digite o caminho de um preset para trocar, ou Enter para sair.")
        while path := input("preset> ").strip():
            try:
                chain.set_params(load_preset(path))
            except (OSError, ValueError, KeyError) as error:
                print(f"Preset não carregado: {error}")

    budget_ms = block_size / SAMPLE_RATE * 1000
    print(f"{stats.blocks} blocos, {stats.xruns} xruns, "
          f"pior bloco {stats.worst_s * 1000:.2f} ms (limite {budget_ms:.2f} ms)")
