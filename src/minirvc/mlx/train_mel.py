from __future__ import annotations

import numpy as np
import mlx.core as mx

from minirvc.train.mel import mel_filter_bank
from minirvc.mlx.train_models import reflect_pad1d_ncl


_MEL_BASIS: dict[tuple[int, int, int, float, float | None], mx.array] = {}
_HANN_WINDOW: dict[int, mx.array] = {}


def spectrogram_np(y: np.ndarray, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, center: bool = False) -> np.ndarray:
    del sampling_rate
    y = np.asarray(y, dtype=np.float32)
    pad = (n_fft - hop_size) // 2
    y = np.pad(y[:, None, :], ((0, 0), (0, 0), (pad, pad)), mode="reflect")[:, 0, :]
    frames = _frames_np(y, n_fft, hop_size)
    window = np.hanning(win_size + 1)[:-1].astype(np.float32)
    if win_size < n_fft:
        padded = np.zeros((n_fft,), dtype=np.float32)
        padded[:win_size] = window
        window = padded
    spec = np.fft.rfft(frames * window, n=n_fft, axis=-1)
    return np.sqrt(spec.real**2 + spec.imag**2 + 1e-6).astype(np.float32).transpose(0, 2, 1)


def spec_to_mel(spec, n_fft: int, num_mels: int, sampling_rate: int, fmin: float, fmax: float | None):
    key = (n_fft, num_mels, sampling_rate, float(fmin), fmax)
    if key not in _MEL_BASIS:
        _MEL_BASIS[key] = mx.array(mel_filter_bank(sampling_rate, n_fft, num_mels, fmin, fmax), dtype=mx.float32)
    return mx.log(mx.maximum(_MEL_BASIS[key] @ spec, 1e-5))


def mel_spectrogram(y, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int, win_size: int, fmin: float, fmax: float | None, center: bool = False):
    spec = spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center)
    return spec_to_mel(spec, n_fft, num_mels, sampling_rate, fmin, fmax)


def spectrogram(y, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, center: bool = False):
    del sampling_rate, center
    if win_size not in _HANN_WINDOW:
        _HANN_WINDOW[win_size] = mx.array(np.hanning(win_size + 1)[:-1].astype(np.float32))
    y = reflect_pad1d_ncl(y[:, None, :], (n_fft - hop_size) // 2, (n_fft - hop_size) // 2)[:, 0, :]
    frames = _frames_mx(y, n_fft, hop_size)
    window = _HANN_WINDOW[win_size]
    if win_size < n_fft:
        window = mx.pad(window, [(0, n_fft - win_size)])
    fft = mx.fft.rfft(frames * window, n=n_fft, axis=-1)
    mag = mx.sqrt(mx.real(fft) ** 2 + mx.imag(fft) ** 2 + 1e-6)
    return mag.swapaxes(1, 2)


def _frames_np(y: np.ndarray, n_fft: int, hop_size: int) -> np.ndarray:
    batch, samples = y.shape
    frame_count = 1 + (samples - n_fft) // hop_size
    shape = (batch, frame_count, n_fft)
    strides = (y.strides[0], hop_size * y.strides[1], y.strides[1])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def _frames_mx(y, n_fft: int, hop_size: int):
    batch, samples = y.shape
    frame_count = 1 + (samples - n_fft) // hop_size
    return mx.as_strided(y, shape=(batch, frame_count, n_fft), strides=(samples, hop_size, 1))

