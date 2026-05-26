from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from minirvc.mlx.weights import as_numpy, conv1d_weight, conv2d_weight, conv_transpose1d_weight, conv_transpose2d_weight, load_torch_checkpoint, validate_weight_keys


def load_pretrained(path: str | Path, model, label: str) -> None:
    checkpoint = load_torch_checkpoint(path)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    expected = {key: value.shape for key, value in tree_flatten(model.parameters())}
    weights: list[tuple[str, np.ndarray]] = []
    for key, raw_value in state.items():
        if key not in expected:
            continue
        value = as_numpy(raw_value)
        value = _coerce_layout(key, value, expected[key])
        weights.append((key, value.astype(np.float32) if value.dtype.kind == "f" else value))
    validate_weight_keys(model, weights, label)
    model.load_weights([(key, mx.array(value)) for key, value in weights], strict=False)
    mx.eval(model.parameters())


def save_training_checkpoint(model, path: str | Path, *, iteration: int, learning_rate: float, epoch: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: _array_for_save(value) for key, value in tree_flatten(model.parameters())}
    arrays["__iteration__"] = np.array(iteration, dtype=np.int64)
    arrays["__learning_rate__"] = np.array(learning_rate, dtype=np.float32)
    arrays["__epoch__"] = np.array(epoch, dtype=np.int64)
    np.savez(path, **arrays)


def load_training_checkpoint(path: str | Path, model, label: str) -> tuple[int, float, int]:
    expected = {key: value.shape for key, value in tree_flatten(model.parameters())}
    weights: list[tuple[str, np.ndarray]] = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("__") or key not in expected:
                continue
            value = np.asarray(data[key])
            if value.shape != expected[key]:
                raise RuntimeError(f"invalid {label} checkpoint shape for {key}: got {value.shape}, expected {expected[key]}")
            weights.append((key, value))
        iteration = int(np.asarray(data["__iteration__"]).reshape(-1)[0]) if "__iteration__" in data.files else 0
        learning_rate = float(np.asarray(data["__learning_rate__"]).reshape(-1)[0]) if "__learning_rate__" in data.files else 0.0
        epoch = int(np.asarray(data["__epoch__"]).reshape(-1)[0]) if "__epoch__" in data.files else 0
    validate_weight_keys(model, weights, label)
    model.load_weights([(key, mx.array(value)) for key, value in weights], strict=False)
    mx.eval(model.parameters())
    return iteration, learning_rate, epoch


def latest_checkpoint_path(directory: str | Path, pattern: str) -> Path | None:
    paths = list(Path(directory).glob(pattern))
    if not paths:
        return None
    return max(paths, key=_checkpoint_step)


def export_small_model(model, path: str | Path, hps, sample_rate: str, use_f0: bool, version: str, epoch: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: _array_for_save(value) for key, value in tree_flatten(model.parameters()) if "enc_q" not in key}
    arrays["__sample_rate__"] = np.array(sample_rate)
    arrays["__f0__"] = np.array(int(use_f0), dtype=np.int64)
    arrays["__version__"] = np.array(version)
    arrays["__epoch__"] = np.array(epoch, dtype=np.int64)
    arrays["__config__"] = np.array(
        [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ],
        dtype=object,
    )
    np.savez(path, **arrays)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)\.mlx\.npz$", path.name)
    return int(match.group(1)) if match else -1


def _array_for_save(value) -> np.ndarray:
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    return np.array(value)


def _coerce_layout(key: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if value.shape == expected_shape:
        return value
    candidates = []
    if value.ndim == 3:
        candidates.extend([conv1d_weight(value), conv_transpose1d_weight(value)])
    elif value.ndim == 4:
        candidates.extend([conv2d_weight(value), conv_transpose2d_weight(value)])
    for candidate in candidates:
        if candidate.shape == expected_shape:
            return candidate
    raise RuntimeError(f"cannot map pretrained weight layout for {key}: got {value.shape}, expected {expected_shape}")
