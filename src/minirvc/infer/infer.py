from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch

from minirvc.infer.pipeline import RvcInferencer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hubert", type=Path, default=Path("assets/hubert/hubert_base.pt"))
    parser.add_argument("--rmvpe", type=Path, default=Path("assets/rmvpe/rmvpe.pt"))
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--index-top-k", type=int, default=8)
    parser.add_argument("--index-query-chunk-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--f0-up-key", type=float, default=0.0)
    parser.add_argument("--f0-threshold", type=float, default=0.03)
    parser.add_argument("--noise-scale", type=float, default=0.66666)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--highpass-hz", type=float, default=48.0)
    parser.add_argument("--max-peak", type=float, default=0.99)
    parser.add_argument("--split-pad-seconds", type=int, default=None)
    parser.add_argument("--split-query-seconds", type=int, default=None)
    parser.add_argument("--split-center-seconds", type=int, default=None)
    parser.add_argument("--split-max-seconds", type=int, default=None)
    parser.add_argument("--no-half", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    inferencer = RvcInferencer(
        voice_path=args.model,
        hubert_path=args.hubert,
        rmvpe_path=args.rmvpe,
        index_path=args.index,
        index_rate=args.index_rate,
        index_top_k=args.index_top_k,
        index_query_chunk_size=args.index_query_chunk_size,
        device=device,
        half=not args.no_half,
    )
    paths = (
        sorted(path for path in args.input.iterdir() if path.suffix.lower() == ".wav")
        if args.input.is_dir()
        else [args.input]
    )
    for path in paths:
        output = args.output / path.name if args.input.is_dir() else args.output
        inferencer.infer_file(
            input_path=path,
            output_path=output,
            sid=args.sid,
            f0_up_key=args.f0_up_key,
            f0_threshold=args.f0_threshold,
            noise_scale=args.noise_scale,
            protect=args.protect,
            highpass_hz=args.highpass_hz,
            max_peak=args.max_peak,
            split_pad_seconds=args.split_pad_seconds,
            split_query_seconds=args.split_query_seconds,
            split_center_seconds=args.split_center_seconds,
            split_max_seconds=args.split_max_seconds,
        )


if __name__ == "__main__":
    main()
