from __future__ import annotations

import argparse
import multiprocessing
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from minirvc.f0.rmvpe import RMVPE
from minirvc.preprocess.audio_io import load_audio_unfiltered


class F0Coarse:
    def __init__(self):
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def __call__(self, f0: np.ndarray) -> np.ndarray:
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1
        return f0_coarse


def extract_f0_directory(
    exp_dir: str | Path,
    model_path: str | Path,
    workers: int,
    device: str | None,
    batch_size: int = 8,
) -> None:
    exp_dir = Path(exp_dir)
    wav_dir = exp_dir / "1_16k_wavs"
    coarse_dir = exp_dir / "2a_f0"
    nsf_dir = exp_dir / "2b-f0nsf"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    nsf_dir.mkdir(parents=True, exist_ok=True)

    files = [path for path in sorted(wav_dir.iterdir()) if path.suffix == ".wav" and "spec" not in str(path)]
    worker_count = max(1, workers)
    tasks = [
        (files[index::worker_count], exp_dir, Path(model_path), device, batch_size)
        for index in range(worker_count)
    ]
    if workers <= 1:
        _extract_part(tasks[0])
        return
    with multiprocessing.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_extract_part, tasks, chunksize=1):
            pass


def _extract_part(task: tuple[list[Path], Path, Path, str | None, int]) -> None:
    files, exp_dir, model_path, device, batch_size = task
    model = RMVPE(model_path, device=device)
    coarse = F0Coarse()
    if batch_size > 1:
        items = [(path, load_audio_unfiltered(path, 16000)) for path in files]
        for group in _keyed_batches(items, batch_size, lambda item: _rmvpe_padded_frames(item[1].shape[0])):
            _extract_f0_batch(group, model, coarse, exp_dir)
        return

    for path in files:
        f0 = model.infer_from_audio(load_audio_unfiltered(path, 16000), threshold=0.03)
        np.save(exp_dir / "2b-f0nsf" / path.name, f0, allow_pickle=False)
        np.save(exp_dir / "2a_f0" / path.name, coarse(f0), allow_pickle=False)


def _extract_f0_batch(
    group: list[tuple[Path, np.ndarray]],
    model: RMVPE,
    coarse: F0Coarse,
    exp_dir: Path,
) -> None:
    with torch.inference_mode():
        mels = []
        frame_lengths = []
        for _, audio in group:
            tensor = torch.from_numpy(audio).float().to(model.device).unsqueeze(0)
            mel = model.mel_extractor(tensor).squeeze(0)
            mels.append(mel)
            frame_lengths.append(mel.shape[-1])

        padded_frames = _next_multiple(max(frame_lengths), 32)
        batch = torch.zeros((len(group), 128, padded_frames), dtype=torch.float32, device=model.device)
        for index, mel in enumerate(mels):
            batch[index, :, : mel.shape[-1]] = mel

        hidden = model.model(batch)[:, : max(frame_lengths)].float().cpu().numpy()

    for index, (path, _) in enumerate(group):
        f0 = model.decode(hidden[index, : frame_lengths[index]], threshold=0.03)
        np.save(exp_dir / "2b-f0nsf" / path.name, f0, allow_pickle=False)
        np.save(exp_dir / "2a_f0" / path.name, coarse(f0), allow_pickle=False)


def _keyed_batches(items: list[tuple[Path, np.ndarray]], batch_size: int, key_fn) -> list[list[tuple[Path, np.ndarray]]]:
    buckets: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    groups: list[list[tuple[Path, np.ndarray]]] = []
    for _, bucket in sorted(buckets.items(), key=lambda pair: pair[0], reverse=True):
        groups.extend(bucket[index : index + batch_size] for index in range(0, len(bucket), batch_size))
    return groups


def _next_multiple(value: int, multiple: int) -> int:
    return multiple * ((value - 1) // multiple + 1)


def _rmvpe_padded_frames(samples: int) -> int:
    return _next_multiple(samples // 160 + 1, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("assets/rmvpe/rmvpe.pt"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extract_f0_directory(args.exp_dir, args.model, args.workers, args.device, args.batch_size)


if __name__ == "__main__":
    main()
