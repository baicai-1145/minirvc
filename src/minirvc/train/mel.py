from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


_MEL_BASIS: dict[tuple[int, int, int, float, float | None, torch.dtype, torch.device], torch.Tensor] = {}
_HANN_WINDOW: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def dynamic_range_compression_torch(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    del sampling_rate
    window_key = (win_size, y.dtype, y.device)
    if window_key not in _HANN_WINDOW:
        _HANN_WINDOW[window_key] = torch.hann_window(win_size, dtype=y.dtype, device=y.device)
    y = F.pad(y.unsqueeze(1), ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_HANN_WINDOW[window_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    key = (n_fft, num_mels, sampling_rate, float(fmin), fmax, spec.dtype, spec.device)
    if key not in _MEL_BASIS:
        mel = mel_filter_bank(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        _MEL_BASIS[key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    return dynamic_range_compression_torch(torch.matmul(_MEL_BASIS[key], spec))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> torch.Tensor:
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)


def mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> np.ndarray:
    fmax = float(sample_rate // 2 if fmax is None else fmax)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    ramps = np.subtract.outer(hz_points, fft_freqs)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    for index in range(n_mels):
        lower = -ramps[index] / (hz_points[index + 1] - hz_points[index])
        upper = ramps[index + 2] / (hz_points[index + 2] - hz_points[index + 1])
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32, copy=False)


def _hz_to_mel(frequencies) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=float)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels = np.array(mels, copy=True)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=float)
    f_sp = 200.0 / 3
    frequencies = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    frequencies = np.array(frequencies, copy=True)
    frequencies[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return frequencies
