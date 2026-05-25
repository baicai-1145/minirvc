from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def build_index(
    exp_dir: str | Path,
    version: str,
    output: str | Path | None = None,
    feature_dir: str | Path | None = None,
    max_vectors: int | None = None,
) -> Path:
    exp_dir = Path(exp_dir)
    feature_dir = Path(feature_dir) if feature_dir is not None else exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    output = Path(output) if output is not None else exp_dir / f"feature_{version}.npz"
    features, files = load_feature_matrix(feature_dir, max_vectors=max_vectors)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        features=features,
        version=np.array(version),
        feature_dim=np.array(features.shape[1], dtype=np.int64),
        files=np.array(files),
    )
    return output


def load_feature_matrix(feature_dir: str | Path, max_vectors: int | None = None) -> tuple[np.ndarray, list[str]]:
    paths = sorted(Path(feature_dir).glob("*.npy"))
    if not paths:
        raise RuntimeError(f"no feature files found in {feature_dir}")
    arrays = []
    files = []
    total = 0
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"feature file must be 2D: {path} shape={array.shape}")
        take = array.shape[0]
        if max_vectors is not None:
            take = min(take, max_vectors - total)
        if take <= 0:
            break
        arrays.append(np.asarray(array[:take], dtype=np.float32))
        files.extend([path.name] * take)
        total += take
    if not arrays:
        raise RuntimeError(f"no feature vectors selected from {feature_dir}")
    return np.concatenate(arrays, axis=0), files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--max-vectors", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = build_index(args.exp_dir, args.version, args.output, args.feature_dir, args.max_vectors)
    data = np.load(output, allow_pickle=False)
    print(f"saved {output} shape={tuple(data['features'].shape)}")


if __name__ == "__main__":
    main()
