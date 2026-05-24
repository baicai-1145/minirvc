from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate = 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        for frame in container.decode(stream):
            chunks.append(frame.to_ndarray().reshape(-1).astype(np.float32))
    audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    return audio, sample_rate


def compare_dirs(old_root: Path, new_root: Path) -> None:
    for subdir in ("0_gt_wavs", "1_16k_wavs"):
        old_files = sorted((old_root / subdir).glob("*.wav"))
        new_files = sorted((new_root / subdir).glob("*.wav"))
        old_names = [path.name for path in old_files]
        new_names = [path.name for path in new_files]
        if old_names != new_names:
            missing = sorted(set(old_names) - set(new_names))
            extra = sorted(set(new_names) - set(old_names))
            raise SystemExit(f"{subdir}: missing={missing[:10]} extra={extra[:10]}")

        max_abs = 0.0
        mean_abs = []
        for old_path in old_files:
            new_path = new_root / subdir / old_path.name
            old_audio, old_sr = read_audio(old_path)
            new_audio, new_sr = read_audio(new_path)
            if old_sr != new_sr or old_audio.shape != new_audio.shape:
                raise SystemExit(
                    f"{subdir}/{old_path.name}: old sr/shape={old_sr}/{old_audio.shape}, "
                    f"new sr/shape={new_sr}/{new_audio.shape}"
                )
            diff = np.abs(old_audio - new_audio)
            max_abs = max(max_abs, float(diff.max(initial=0)))
            mean_abs.append(float(diff.mean()) if diff.size else 0.0)
        print(
            f"{subdir}: files={len(old_files)} max_abs={max_abs:.9g} "
            f"mean_abs={sum(mean_abs) / max(1, len(mean_abs)):.9g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    args = parser.parse_args()
    compare_dirs(args.old_root, args.new_root)


if __name__ == "__main__":
    main()
