from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from minirvc.preprocess.audio_io import load_audio_unfiltered


def extract_hubert_directory(
    exp_dir: str | Path,
    model_path: str | Path,
    version: str,
    device: str | None,
    batch_size: int = 16,
    backend: str = "torch",
) -> None:
    if backend == "mlx":
        from minirvc.mlx.hubert import extract_hubert_directory_mlx

        extract_hubert_directory_mlx(exp_dir, model_path, version, device, batch_size)
        return
    if backend != "torch":
        raise ValueError("backend must be 'torch' or 'mlx'")

    import torch

    from minirvc.content.hubert import load_hubert

    exp_dir = Path(exp_dir)
    wav_dir = exp_dir / "1_16k_wavs"
    out_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = load_hubert(model_path, device=device)
    files = [path for path in sorted(wav_dir.iterdir()) if path.suffix == ".wav"]

    if batch_size > 1:
        items = [(path, load_audio_unfiltered(path, 16000)) for path in files]
        for group in _keyed_batches(items, batch_size, lambda item: item[1].shape[0]):
            _extract_hubert_batch(group, model, version, out_dir, device)
        return

    for wav_path in files:
        out_path = out_dir / wav_path.name.replace(".wav", ".npy")
        audio = torch.from_numpy(load_audio_unfiltered(wav_path, 16000)).float().view(1, -1).to(device)
        with torch.inference_mode():
            features = model.infer(audio, version=version)
        array = features.squeeze(0).float().cpu().numpy()
        np.save(out_path, array, allow_pickle=False)


def _extract_hubert_batch(
    group: list[tuple[Path, np.ndarray]],
    model: Any,
    version: str,
    out_dir: Path,
    device: Any,
) -> None:
    import torch

    sample_lengths = [audio.shape[0] for _, audio in group]
    max_samples = max(sample_lengths)
    audio_batch = torch.zeros((len(group), max_samples), dtype=torch.float32, device=device)
    for index, (_, audio) in enumerate(group):
        audio_batch[index, : audio.shape[0]] = torch.from_numpy(audio).float().to(device)

    with torch.inference_mode():
        arrays = model.infer(audio_batch, version=version).float().cpu().numpy()

    for index, (path, _) in enumerate(group):
        np.save(out_dir / path.name.replace(".wav", ".npy"), arrays[index], allow_pickle=False)


def _keyed_batches(items: list[tuple[Path, np.ndarray]], batch_size: int, key_fn) -> list[list[tuple[Path, np.ndarray]]]:
    buckets: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    groups: list[list[tuple[Path, np.ndarray]]] = []
    for _, bucket in sorted(buckets.items(), key=lambda pair: pair[0], reverse=True):
        groups.extend(bucket[index : index + batch_size] for index in range(0, len(bucket), batch_size))
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("assets/hubert/hubert_base.pt"))
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extract_hubert_directory(args.exp_dir, args.model, args.version, args.device, args.batch_size, args.backend)


if __name__ == "__main__":
    main()
