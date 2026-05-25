from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from minirvc.train.nn.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


@dataclass(frozen=True)
class LoadedVoice:
    model: torch.nn.Module
    sample_rate: int
    use_f0: bool
    version: str
    use_half: bool


def load_voice_model(path: str | Path, device: str | torch.device, half: bool = True) -> LoadedVoice:
    device = torch.device(device)
    use_half = bool(half and device.type == "cuda")
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    config = list(checkpoint["config"])
    version = str(checkpoint["version"])
    use_f0 = bool(checkpoint["f0"])
    sample_rate = int(config[-1])
    cls = {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }[(version, use_f0)]
    model = cls(*config, is_half=use_half)
    missing, unexpected = model.load_state_dict(checkpoint["weight"], strict=False)
    unexpected = list(unexpected)
    missing = [key for key in missing if not key.startswith("enc_q.")]
    if missing or unexpected:
        raise RuntimeError(f"voice checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    if use_half:
        model.half()
    else:
        model.float()
    return LoadedVoice(model=model, sample_rate=sample_rate, use_f0=use_f0, version=version, use_half=use_half)
