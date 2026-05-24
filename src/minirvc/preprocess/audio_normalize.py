from __future__ import annotations

import numpy as np


def normalize_clip(
    audio: np.ndarray,
    max_amplitude: float = 0.9,
    blend: float = 0.75,
    reject_peak: float = 2.5,
) -> np.ndarray | None:
    peak = float(np.abs(audio).max())
    if peak > reject_peak:
        return None
    normalized = audio / peak * (max_amplitude * blend) + (1.0 - blend) * audio
    return normalized.astype(np.float32, copy=False)
