from __future__ import annotations

import numpy as np


def split_by_silence(
    waveform: np.ndarray,
    sample_rate: int,
    threshold_db: float = -42.0,
    min_length_ms: int = 1500,
    min_interval_ms: int = 400,
    hop_size_ms: int = 15,
    max_silence_kept_ms: int = 500,
) -> list[np.ndarray]:
    samples = np.asarray(waveform)
    hop_size = round(sample_rate * hop_size_ms / 1000)
    win_size = min(round(sample_rate * min_interval_ms / 1000), 4 * hop_size)
    min_length = round(sample_rate * min_length_ms / 1000 / hop_size)
    min_interval = round(sample_rate * min_interval_ms / 1000 / hop_size)
    max_silence_kept = round(sample_rate * max_silence_kept_ms / 1000 / hop_size)

    if samples.shape[0] <= min_length:
        return [samples]

    rms = frame_rms(samples, frame_length=win_size, hop_length=hop_size)
    threshold = 10 ** (threshold_db / 20.0)
    silence_tags: list[tuple[int, int]] = []
    silence_start = None
    clip_start = 0

    for index, value in enumerate(rms):
        if value < threshold:
            if silence_start is None:
                silence_start = index
            continue
        if silence_start is None:
            continue

        is_leading_silence = silence_start == 0 and index > max_silence_kept
        need_slice_middle = (
            index - silence_start >= min_interval
            and index - clip_start >= min_length
        )
        if not is_leading_silence and not need_slice_middle:
            silence_start = None
            continue

        if index - silence_start <= max_silence_kept:
            position = rms[silence_start : index + 1].argmin() + silence_start
            silence_tags.append((0, position) if silence_start == 0 else (position, position))
            clip_start = position
        elif index - silence_start <= max_silence_kept * 2:
            position = rms[index - max_silence_kept : silence_start + max_silence_kept + 1].argmin()
            position += index - max_silence_kept
            left = rms[silence_start : silence_start + max_silence_kept + 1].argmin()
            left += silence_start
            right = rms[index - max_silence_kept : index + 1].argmin()
            right += index - max_silence_kept
            if silence_start == 0:
                silence_tags.append((0, right))
                clip_start = right
            else:
                silence_tags.append((min(left, position), max(right, position)))
                clip_start = max(right, position)
        else:
            left = rms[silence_start : silence_start + max_silence_kept + 1].argmin()
            left += silence_start
            right = rms[index - max_silence_kept : index + 1].argmin()
            right += index - max_silence_kept
            silence_tags.append((0, right) if silence_start == 0 else (left, right))
            clip_start = right

        silence_start = None

    total_frames = rms.shape[0]
    if silence_start is not None and total_frames - silence_start >= min_interval:
        silence_end = min(total_frames, silence_start + max_silence_kept)
        position = rms[silence_start : silence_end + 1].argmin() + silence_start
        silence_tags.append((position, total_frames + 1))

    if not silence_tags:
        return [samples]

    chunks: list[np.ndarray] = []
    if silence_tags[0][0] > 0:
        chunks.append(_slice_frames(samples, hop_size, 0, silence_tags[0][0]))
    for index in range(len(silence_tags) - 1):
        chunks.append(_slice_frames(samples, hop_size, silence_tags[index][1], silence_tags[index + 1][0]))
    if silence_tags[-1][1] < total_frames:
        chunks.append(_slice_frames(samples, hop_size, silence_tags[-1][1], total_frames))
    return chunks


def split_fixed_length(
    waveform: np.ndarray,
    sample_rate: int,
    seconds: float = 3.7,
    overlap_seconds: float = 0.3,
) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    tail_seconds = seconds + overlap_seconds
    start_step = int(sample_rate * (seconds - overlap_seconds))
    chunk_length = int(seconds * sample_rate)
    tail_length = int(tail_seconds * sample_rate)
    index = 0
    while True:
        start = start_step * index
        index += 1
        if len(waveform[start:]) > tail_length:
            chunks.append(waveform[start : start + chunk_length])
        else:
            chunks.append(waveform[start:])
            break
    return chunks


def iter_legacy_fixed_length(
    waveform: np.ndarray,
    sample_rate: int,
    seconds: float = 3.7,
    overlap_seconds: float = 0.3,
):
    tail_seconds = seconds + overlap_seconds
    start_step = int(sample_rate * (seconds - overlap_seconds))
    chunk_length = int(seconds * sample_rate)
    tail_length = int(tail_seconds * sample_rate)
    index = 0
    while True:
        start = start_step * index
        index += 1
        if len(waveform[start:]) > tail_length:
            yield waveform[start : start + chunk_length], 0, 1
        else:
            yield waveform[start:], 1, 0
            break


def frame_rms(
    y: np.ndarray, frame_length: int = 2048, hop_length: int = 512
) -> np.ndarray:
    y = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    shape = (y.shape[0] - frame_length + 1, frame_length)
    strides = (y.strides[0], y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    frames = frames[::hop_length]
    return np.sqrt(np.mean(np.abs(frames) ** 2, axis=1))


def _slice_frames(waveform: np.ndarray, hop_size: int, begin: int, end: int) -> np.ndarray:
    return waveform[begin * hop_size : min(waveform.shape[0], end * hop_size)]
