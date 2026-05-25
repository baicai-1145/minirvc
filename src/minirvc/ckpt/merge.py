from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


def merge_models(model_a: str | Path, model_b: str | Path, output: str | Path, alpha: float) -> Path:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    a = torch.load(model_a, map_location="cpu", weights_only=False)
    b = torch.load(model_b, map_location="cpu", weights_only=False)
    _validate_compatible(a, b)
    out: dict[str, Any] = OrderedDict()
    out["weight"] = OrderedDict()
    for key, value_a in a["weight"].items():
        value_b = b["weight"][key]
        if torch.is_floating_point(value_a):
            merged = value_a.float() * float(alpha) + value_b.float() * float(1.0 - alpha)
            out["weight"][key] = merged.to(dtype=value_a.dtype)
        else:
            if not torch.equal(value_a, value_b):
                raise ValueError(f"non-floating weight differs: {key}")
            out["weight"][key] = value_a
    out["config"] = a["config"]
    out["sr"] = a["sr"]
    out["f0"] = a["f0"]
    out["version"] = a["version"]
    out["info"] = f"merged alpha={alpha:g} a={Path(model_a).name} b={Path(model_b).name}"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    return output


def _validate_compatible(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in ("config", "sr", "f0", "version"):
        if a[key] != b[key]:
            raise ValueError(f"model metadata differs: {key}")
    keys_a = list(a["weight"].keys())
    keys_b = list(b["weight"].keys())
    if keys_a != keys_b:
        raise ValueError("model weight keys differ")
    for key in keys_a:
        if a["weight"][key].shape != b["weight"][key].shape:
            raise ValueError(f"weight shape differs: {key}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = merge_models(args.model_a, args.model_b, args.output, args.alpha)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
