from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def latest_checkpoint_path(directory: str | Path, pattern: str) -> Path | None:
    paths = sorted(Path(directory).glob(pattern), key=_checkpoint_number)
    return paths[-1] if paths else None


def load_training_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, load_optimizer: bool = True):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(_compatible_state_dict(model, checkpoint["model"]), strict=False)
    if optimizer is not None and load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["learning_rate"], int(checkpoint["iteration"])


def load_pretrained(path: str | Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        path,
    )


def export_small_model(
    model: torch.nn.Module,
    output_path: str | Path,
    hps,
    sample_rate: str,
    use_f0: bool,
    version: str,
    epoch: int,
) -> None:
    opt: dict[str, Any] = OrderedDict()
    opt["weight"] = OrderedDict()
    for key, value in model.state_dict().items():
        if "enc_q" not in key:
            opt["weight"][key] = value.detach().cpu().half()
    opt["config"] = [
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
    ]
    opt["info"] = f"{epoch}epoch"
    opt["sr"] = sample_rate
    opt["f0"] = int(use_f0)
    opt["version"] = version
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(opt, output_path)


def _compatible_state_dict(model: torch.nn.Module, saved: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = model.state_dict()
    return {key: saved[key] if key in saved and saved[key].shape == value.shape else value for key, value in current.items()}


def _checkpoint_number(path: Path) -> int:
    match = re.findall(r"\d+", path.name)
    return int(match[-1]) if match else -1
