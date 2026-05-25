from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch


def model_info(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["weight"]
    return {
        "path": str(path),
        "version": checkpoint["version"],
        "sample_rate": checkpoint["sr"],
        "f0": bool(checkpoint["f0"]),
        "info": checkpoint.get("info", ""),
        "config": checkpoint["config"],
        "weight_count": len(weights),
        "parameter_count": int(sum(value.numel() for value in weights.values() if torch.is_tensor(value))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(model_info(args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
