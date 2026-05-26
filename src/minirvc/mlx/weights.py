from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


def load_torch_checkpoint(path: str | Path, *, fairseq: bool = False) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"loading PyTorch checkpoint {path} requires torch; convert it to an MLX .npz first "
            "or run this command in the regular minirvc environment"
        ) from exc

    if not fairseq:
        return torch.load(path, map_location="cpu", weights_only=False)

    class Dictionary:
        pass

    fairseq_module = types.ModuleType("fairseq")
    data_module = types.ModuleType("fairseq.data")
    dictionary_module = types.ModuleType("fairseq.data.dictionary")
    dictionary_module.Dictionary = Dictionary
    original = {name: sys.modules.get(name) for name in ("fairseq", "fairseq.data", "fairseq.data.dictionary")}
    sys.modules["fairseq"] = fairseq_module
    sys.modules["fairseq.data"] = data_module
    sys.modules["fairseq.data.dictionary"] = dictionary_module
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_npz_weights(path: str | Path) -> list[tuple[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        return [(key, np.asarray(data[key])) for key in data.files]


def save_npz_weights(path: str | Path, weights: list[tuple[str, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: value for key, value in weights})


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "float") and getattr(value, "is_floating_point", lambda: False)():
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def conv1d_weight(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).transpose(0, 2, 1)


def conv2d_weight(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).transpose(0, 2, 3, 1)


def conv_transpose1d_weight(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).transpose(1, 2, 0)


def conv_transpose2d_weight(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).transpose(1, 2, 3, 0)


def float_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def validate_weight_keys(module, weights: list[tuple[str, np.ndarray]], label: str) -> None:
    from mlx.utils import tree_flatten

    expected = {key for key, _ in tree_flatten(module.parameters())}
    actual = {key for key, _ in weights}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(f"{label} MLX weight mismatch: missing={missing}, unexpected={unexpected}")
