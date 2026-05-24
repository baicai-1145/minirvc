from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import soxr
from scipy import signal


def load_audio(path: str | Path, sample_rate: int, highpass_hz: float = 48.0) -> np.ndarray:
    path = Path(path)
    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))

        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return np.empty(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if highpass_hz <= 0:
        return audio
    b, a = signal.butter(N=5, Wn=highpass_hz, btype="high", fs=sample_rate)
    return signal.lfilter(b, a, audio)


def load_audio_unfiltered(path: str | Path, sample_rate: int) -> np.ndarray:
    return load_audio(path, sample_rate, highpass_hz=0.0)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)

    with av.open(str(path), mode="w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    resampled = soxr.resample(audio, in_rate=src_rate, out_rate=dst_rate, quality="soxr_hq")
    target_length = int(np.ceil(audio.shape[-1] * float(dst_rate) / src_rate))
    if resampled.shape[-1] > target_length:
        resampled = resampled[:target_length]
    elif resampled.shape[-1] < target_length:
        resampled = np.pad(resampled, (0, target_length - resampled.shape[-1]))
    return np.asarray(resampled, dtype=audio.dtype)
