from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SAMPLE_RATE_NUMERIC = {"32k": 32000, "40k": 40000, "48k": 48000}


_BASE_TRAIN = {
    "log_interval": 200,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 4,
    "fp16_run": True,
    "lr_decay": 0.999875,
    "segment_size": 12800,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
}


_BASE_MODEL = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_initial_channel": 512,
    "use_spectral_norm": False,
    "gin_channels": 256,
    "spk_embed_dim": 109,
}


_RATE_CONFIGS = {
    ("v1", "32k"): {
        "data": {"sampling_rate": 32000, "filter_length": 1024, "hop_length": 320, "win_length": 1024, "n_mel_channels": 80},
        "model": {"upsample_rates": [10, 4, 2, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4, 4]},
    },
    ("v1", "40k"): {
        "data": {"sampling_rate": 40000, "filter_length": 2048, "hop_length": 400, "win_length": 2048, "n_mel_channels": 125},
        "model": {"upsample_rates": [10, 10, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4]},
    },
    ("v1", "48k"): {
        "data": {"sampling_rate": 48000, "filter_length": 2048, "hop_length": 480, "win_length": 2048, "n_mel_channels": 128},
        "model": {"upsample_rates": [10, 6, 2, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4, 4]},
    },
    ("v2", "32k"): {
        "data": {"sampling_rate": 32000, "filter_length": 1024, "hop_length": 320, "win_length": 1024, "n_mel_channels": 80},
        "model": {"upsample_rates": [10, 8, 2, 2], "upsample_kernel_sizes": [20, 16, 4, 4]},
    },
    ("v2", "40k"): {
        "train": {"log_interval": 50, "lr_decay": 0.99975, "warmup_epochs": 100},
        "data": {"sampling_rate": 40000, "filter_length": 2048, "hop_length": 400, "win_length": 2048, "n_mel_channels": 125},
        "model": {"upsample_rates": [10, 10, 2, 2], "upsample_kernel_sizes": [16, 16, 4, 4]},
    },
    ("v2", "48k"): {
        "data": {"sampling_rate": 48000, "filter_length": 2048, "hop_length": 480, "win_length": 2048, "n_mel_channels": 128},
        "model": {"upsample_rates": [12, 10, 2, 2], "upsample_kernel_sizes": [24, 20, 4, 4]},
    },
}


def default_config(version: str, sample_rate: str) -> dict[str, Any]:
    key = (version, sample_rate)
    if key not in _RATE_CONFIGS:
        raise ValueError(f"unsupported version/sample-rate: {version} {sample_rate}")
    config = {
        "train": copy.deepcopy(_BASE_TRAIN),
        "data": {
            "max_wav_value": 32768.0,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        },
        "model": copy.deepcopy(_BASE_MODEL),
    }
    _deep_update(config, copy.deepcopy(_RATE_CONFIGS[key]))
    return config


def load_config(version: str, sample_rate: str, path: str | Path | None = None) -> dict[str, Any]:
    config = default_config(version, sample_rate)
    if path is not None:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        _deep_update(config, overrides)
    return config


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
