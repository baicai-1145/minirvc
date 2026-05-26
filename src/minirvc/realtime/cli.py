from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import time

import numpy as np

from minirvc.preprocess.audio_io import load_audio, write_wav
from minirvc.realtime.engine import RealtimeConfig, create_realtime_engine
from minirvc.realtime.stream import list_devices, run_stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hubert", type=Path, default=Path("assets/hubert/hubert_base.pt"))
    parser.add_argument("--rmvpe", type=Path, default=Path("assets/rmvpe/rmvpe.pt"))
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--index-top-k", type=int, default=8)
    parser.add_argument("--index-query-chunk-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--f0-up-key", type=float, default=0.0)
    parser.add_argument("--f0-threshold", type=float, default=0.03)
    parser.add_argument("--noise-scale", type=float, default=0.66666)
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument("--sola-search-time", type=float, default=0.01)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--offline-input", type=Path, default=None)
    parser.add_argument("--offline-output", type=Path, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        list_devices()
        return
    if args.model is None:
        raise SystemExit("--model is required")
    config = RealtimeConfig(
        voice_path=args.model,
        hubert_path=args.hubert,
        rmvpe_path=args.rmvpe,
        index_path=args.index,
        index_rate=args.index_rate,
        index_top_k=args.index_top_k,
        index_query_chunk_size=args.index_query_chunk_size,
        backend=args.backend,
        device=args.device,
        half=not args.no_half,
        precision=args.precision,
        sid=args.sid,
        f0_up_key=args.f0_up_key,
        f0_threshold=args.f0_threshold,
        noise_scale=args.noise_scale,
        block_time=args.block_time,
        crossfade_time=args.crossfade_time,
        extra_time=args.extra_time,
        sola_search_time=args.sola_search_time,
    )
    engine = create_realtime_engine(config)
    if args.offline_input is not None:
        if args.offline_output is None:
            raise SystemExit("--offline-output is required with --offline-input")
        _run_offline(engine, args.offline_input, args.offline_output)
        return
    run_stream(
        engine,
        input_device=_device_arg(args.input_device),
        output_device=_device_arg(args.output_device),
        channels=args.channels,
        queue_size=args.queue_size,
    )


def _run_offline(engine, input_path: Path, output_path: Path) -> None:
    audio = load_audio(input_path, engine.sample_rate, highpass_hz=0.0)
    original_length = audio.shape[0]
    pad = (-audio.shape[0]) % engine.block_frame
    if pad:
        audio = np.pad(audio, (0, pad))
    output = []
    started = time.perf_counter()
    for start in range(0, audio.shape[0], engine.block_frame):
        output.append(engine.process_block(audio[start : start + engine.block_frame]))
    converted = np.concatenate(output)[:original_length]
    write_wav(output_path, converted, engine.sample_rate)
    elapsed = time.perf_counter() - started
    audio_seconds = original_length / engine.sample_rate
    stats = engine.last_stats
    print(
        "processed %.2fs audio in %.2fs, last block %.1fms "
        "(feature %.1f, index %.1f, f0 %.1f, model %.1f)"
        % (
            audio_seconds,
            elapsed,
            stats.total_s * 1000,
            stats.feature_s * 1000,
            stats.index_s * 1000,
            stats.f0_s * 1000,
            stats.model_s * 1000,
        )
    )


def _device_arg(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
