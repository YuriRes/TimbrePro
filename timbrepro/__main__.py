"""Linha de comando: python -m timbrepro {render,live,bench}."""

import argparse
import time

import numpy as np
from scipy.signal import resample_poly

from timbrepro.chain import FxChain
from timbrepro.params import BLOCK_SIZE, SAMPLE_RATE, load_preset
from timbrepro.synth import synth_di

DEFAULT_PRESET = "presets/crunch.json"
TAIL_S = 3  # silêncio no fim do render, para ouvir a cauda do delay e do reverb


def read_mono(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    return audio if sr == SAMPLE_RATE else resample_poly(audio, SAMPLE_RATE, sr)


def cmd_render(args):
    import soundfile as sf

    dry = read_mono(args.input) if args.input else synth_di()
    wet = FxChain(load_preset(args.preset)).process(
        np.concatenate([dry, np.zeros(TAIL_S * SAMPLE_RATE)]))
    sf.write(args.output, wet, SAMPLE_RATE, subtype="PCM_24")
    print(f"{args.output}: {len(wet) / SAMPLE_RATE:.1f} s, pico {np.abs(wet).max():.2f}")


def cmd_live(args):
    from timbrepro.live import run

    run(args.preset, args.block, args.device, args.bypass)


def cmd_bench(args):
    chain = FxChain(load_preset(args.preset))
    audio = synth_di()
    blocks = [audio[i:i + args.block] for i in range(0, len(audio) - args.block, args.block)]
    times_ms = []
    for block in blocks:
        start = time.perf_counter()
        chain.process(block)
        times_ms.append((time.perf_counter() - start) * 1000)
    budget_ms = args.block / SAMPLE_RATE * 1000
    print(f"{len(blocks)} blocos de {args.block} amostras (limite {budget_ms:.2f} ms por bloco)")
    print(f"média {np.mean(times_ms):.3f} ms, p99 {np.percentile(times_ms, 99):.3f} ms, "
          f"pior {np.max(times_ms):.3f} ms")


def device_id(value):
    """Número ou nome do dispositivo; "12,10" separa entrada e saída."""
    ids = [int(v) if v.isdigit() else v for v in value.split(",")]
    return ids[0] if len(ids) == 1 else tuple(ids)


def main():
    parser = argparse.ArgumentParser(prog="timbrepro", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render", help="aplica um preset a um arquivo de áudio")
    render.add_argument("--input", help="áudio de entrada; sem isto, usa um riff sintético")
    render.add_argument("--output", default="out.wav")
    render.add_argument("--preset", default=DEFAULT_PRESET)
    render.set_defaults(func=cmd_render)

    live = commands.add_parser("live", help="toca em tempo real pela interface de áudio")
    live.add_argument("--preset", default=DEFAULT_PRESET)
    live.add_argument("--block", type=int, default=BLOCK_SIZE, help="amostras por bloco")
    live.add_argument("--device", type=device_id,
                      help='número ou nome do dispositivo; "12,10" separa entrada e saída')
    live.add_argument("--bypass", action="store_true", help="passthrough, sem efeitos")
    live.set_defaults(func=cmd_live)

    bench = commands.add_parser("bench", help="mede o tempo por bloco, sem precisar de áudio")
    bench.add_argument("--preset", default=DEFAULT_PRESET)
    bench.add_argument("--block", type=int, default=BLOCK_SIZE, help="amostras por bloco")
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
