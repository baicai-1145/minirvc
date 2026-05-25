from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch

from minirvc.train.trainer import train


def infer_pretrained_paths(version: str, sample_rate: str, use_f0: bool) -> tuple[Path, Path]:
    root = Path("assets/pretrained_v2" if version == "v2" else "assets/pretrained")
    prefix = "f0" if use_f0 else ""
    return root / f"{prefix}G{sample_rate}.pth", root / f"{prefix}D{sample_rate}.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--filelist", type=Path, default=None)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--sample-rate", choices=("32k", "40k", "48k"), required=True)
    parser.add_argument("--f0", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save-every-epoch", type=int, default=5)
    parser.add_argument("--pretrain-g", type=Path, default=None)
    parser.add_argument("--pretrain-d", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-export-final", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    filelist = args.filelist or args.exp_dir / "filelist.txt"
    pretrain_g, pretrain_d = infer_pretrained_paths(args.version, args.sample_rate, args.f0)
    if args.pretrain_g is not None:
        pretrain_g = args.pretrain_g
    if args.pretrain_d is not None:
        pretrain_d = args.pretrain_d

    if args.device is None and torch.cuda.is_available():
        args.device = "cuda:0"

    train(
        exp_dir=args.exp_dir,
        filelist=filelist,
        version=args.version,
        sample_rate=args.sample_rate,
        use_f0=args.f0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_epoch=args.save_every_epoch,
        pretrain_g=pretrain_g,
        pretrain_d=pretrain_d,
        device=args.device,
        config_json=args.config_json,
        num_workers=args.workers,
        resume=not args.no_resume,
        export_final=not args.no_export_final,
    )


if __name__ == "__main__":
    main()
