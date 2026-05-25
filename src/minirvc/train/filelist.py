from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def build_filelist(
    dataset_dir: str | Path,
    version: str,
    sample_rate: str,
    use_f0: bool,
    output: str | Path | None = None,
    mute_root: str | Path | None = None,
    sid: int = 0,
) -> Path:
    dataset_dir = Path(dataset_dir)
    gt_dir = dataset_dir / "0_gt_wavs"
    feature_dir = dataset_dir / ("3_feature256" if version == "v1" else "3_feature768")
    f0_dir = dataset_dir / "2a_f0"
    f0nsf_dir = dataset_dir / "2b-f0nsf"
    if output is None:
        output = dataset_dir / "filelist.txt"
    output = Path(output)

    gt_names = {_stem(path) for path in gt_dir.iterdir() if path.suffix == ".wav"}
    feature_names = {_stem(path) for path in feature_dir.iterdir() if path.suffix == ".npy"}
    if use_f0:
        f0_names = {_stem(path) for path in f0_dir.iterdir() if path.suffix == ".npy"}
        f0nsf_names = {_stem(path) for path in f0nsf_dir.iterdir() if path.suffix == ".npy"}
        names = sorted(gt_names & feature_names & f0_names & f0nsf_names)
    else:
        names = sorted(gt_names & feature_names)
    if not names:
        raise RuntimeError(f"no training samples found in {dataset_dir}")

    lines = []
    for name in names:
        if use_f0:
            lines.append(
                f"{gt_dir.resolve()}/{name}.wav|{feature_dir.resolve()}/{name}.npy|"
                f"{f0_dir.resolve()}/{name}.wav.npy|{f0nsf_dir.resolve()}/{name}.wav.npy|{sid}"
            )
        else:
            lines.append(f"{gt_dir.resolve()}/{name}.wav|{feature_dir.resolve()}/{name}.npy|{sid}")

    if mute_root is not None:
        mute_lines = _mute_lines(Path(mute_root), version, sample_rate, use_f0, sid)
        lines.extend(mute_lines)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _mute_lines(mute_root: Path, version: str, sample_rate: str, use_f0: bool, sid: int) -> list[str]:
    feature_dim = 256 if version == "v1" else 768
    wav = mute_root / "0_gt_wavs" / f"mute{sample_rate}.wav"
    feature = mute_root / f"3_feature{feature_dim}" / "mute.npy"
    if use_f0:
        f0 = mute_root / "2a_f0" / "mute.wav.npy"
        f0nsf = mute_root / "2b-f0nsf" / "mute.wav.npy"
        required = (wav, feature, f0, f0nsf)
    else:
        required = (wav, feature)
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"missing mute sample files under {mute_root}")
    if use_f0:
        return [f"{wav.resolve()}|{feature.resolve()}|{f0.resolve()}|{f0nsf.resolve()}|{sid}" for _ in range(2)]
    return [f"{wav.resolve()}|{feature.resolve()}|{sid}" for _ in range(2)]


def _stem(path: Path) -> str:
    return path.name.split(".")[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--sample-rate", choices=("32k", "40k", "48k"), required=True)
    parser.add_argument("--f0", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mute-root", type=Path, default=None)
    parser.add_argument("--sid", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = build_filelist(
        dataset_dir=args.dataset_dir,
        version=args.version,
        sample_rate=args.sample_rate,
        use_f0=args.f0,
        output=args.output,
        mute_root=args.mute_root,
        sid=args.sid,
    )
    print(path)


if __name__ == "__main__":
    main()
