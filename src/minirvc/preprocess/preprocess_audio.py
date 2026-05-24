from __future__ import annotations

import argparse
import multiprocessing
from collections.abc import Iterable
from pathlib import Path

from minirvc.preprocess.audio_io import load_audio, resample_audio, write_wav
from minirvc.preprocess.audio_normalize import normalize_clip
from minirvc.preprocess.audio_slice import iter_legacy_fixed_length, split_by_silence


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int,
    workers: int,
    seconds: float = 3.7,
    highpass_hz: float = 48.0,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    gt_dir = output_dir / "0_gt_wavs"
    wav16k_dir = output_dir / "1_16k_wavs"
    gt_dir.mkdir(parents=True, exist_ok=True)
    wav16k_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (path, index, gt_dir, wav16k_dir, sample_rate, seconds, highpass_hz)
        for index, path in enumerate(sorted(input_dir.iterdir()))
        if path.is_file()
    ]

    if workers <= 1:
        for task in tasks:
            _preprocess_one(task)
        return

    with multiprocessing.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_preprocess_one, tasks, chunksize=4):
            pass


def _preprocess_one(task: tuple[Path, int, Path, Path, int, float, float]) -> None:
    path, file_index, gt_dir, wav16k_dir, sample_rate, seconds, highpass_hz = task
    audio = load_audio(path, sample_rate, highpass_hz=highpass_hz)
    clip_index = 0
    for sliced in split_by_silence(audio, sample_rate):
        for clip, index_delta_before_write, index_delta_after_write in iter_legacy_fixed_length(
            sliced, sample_rate, seconds=seconds
        ):
            clip_index += index_delta_before_write
            normalized = normalize_clip(clip)
            if normalized is None:
                continue
            name = f"{file_index}_{clip_index}.wav"
            write_wav(gt_dir / name, normalized, sample_rate)
            write_wav(wav16k_dir / name, resample_audio(normalized, sample_rate, 16000), 16000)
            clip_index += index_delta_after_write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=3.7)
    parser.add_argument("--highpass-hz", type=float, default=48.0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        workers=args.workers,
        seconds=args.seconds,
        highpass_hz=args.highpass_hz,
    )


if __name__ == "__main__":
    main()
