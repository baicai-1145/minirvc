from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minirvc.mlx.train_mel import spectrogram_np
from minirvc.preprocess.audio_io import load_audio_unfiltered


@dataclass(frozen=True)
class TrainItem:
    wav_path: Path
    feature_path: Path
    sid: int
    f0_path: Path | None = None
    f0nsf_path: Path | None = None


class RvcMlxDataset:
    def __init__(self, filelist: str | Path, hps, use_f0: bool):
        self.items = load_filelist(filelist, use_f0)
        self.use_f0 = bool(use_f0)
        self.sampling_rate = hps.sampling_rate
        self.filter_length = hps.filter_length
        self.hop_length = hps.hop_length
        self.win_length = hps.win_length
        self.min_text_len = getattr(hps, "min_text_len", 1)
        self.max_text_len = getattr(hps, "max_text_len", 5000)
        self.items, self.lengths = self._filter_items()

    def _filter_items(self) -> tuple[list[TrainItem], list[int]]:
        filtered = []
        lengths = []
        for item in self.items:
            feature_len = int(np.load(item.feature_path, mmap_mode="r", allow_pickle=False).shape[0])
            if self.min_text_len <= feature_len <= self.max_text_len:
                filtered.append(item)
                lengths.append(item.wav_path.stat().st_size // (3 * self.hop_length))
        if not filtered:
            raise RuntimeError("no training samples after length filtering")
        return filtered, lengths

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        phone = _load_phone(item.feature_path)
        audio = load_audio_unfiltered(item.wav_path, self.sampling_rate).astype(np.float32, copy=False)
        wav = audio[None, :]
        spec = spectrogram_np(wav, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)[0]
        sid = np.array([item.sid], dtype=np.int64)
        if self.use_f0:
            assert item.f0_path is not None and item.f0nsf_path is not None
            pitch = np.load(item.f0_path, allow_pickle=False).astype(np.int64)
            pitchf = np.load(item.f0nsf_path, allow_pickle=False).astype(np.float32)
            return _trim_f0(spec, wav, phone, pitch, pitchf, self.hop_length) + (sid,)
        return _trim_nof0(spec, wav, phone, self.hop_length) + (sid,)

    def batches(self, batch_size: int, epoch: int, boundaries: list[int] | None = None):
        boundaries = boundaries or [100, 200, 300, 400, 500, 600, 700, 800, 900]
        rng = np.random.default_rng(epoch)
        buckets = _create_buckets(self.lengths, boundaries)
        batches = []
        for bucket in buckets:
            order = rng.permutation(len(bucket)).tolist()
            ordered = [bucket[index] for index in order]
            rem = (batch_size - (len(ordered) % batch_size)) % batch_size
            if rem:
                ordered += (ordered * math.ceil(rem / len(ordered)))[:rem]
            batches.extend(ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size))
        order = rng.permutation(len(batches)).tolist()
        for batch_index in order:
            yield collate([self[index] for index in batches[batch_index]], self.use_f0)


def load_filelist(path: str | Path, use_f0: bool) -> list[TrainItem]:
    items = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split("|")
        if use_f0:
            if len(fields) != 5:
                raise ValueError(f"invalid f0 filelist row: {line}")
            items.append(TrainItem(Path(fields[0]), Path(fields[1]), int(fields[4]), Path(fields[2]), Path(fields[3])))
        else:
            if len(fields) != 3:
                raise ValueError(f"invalid filelist row: {line}")
            items.append(TrainItem(Path(fields[0]), Path(fields[1]), int(fields[2])))
    if not items:
        raise RuntimeError(f"empty filelist: {path}")
    return items


def collate(batch, use_f0: bool):
    sorted_ids = sorted(range(len(batch)), key=lambda index: batch[index][0].shape[1], reverse=True)
    spec_padded, spec_lengths = _pad([batch[index][0] for index in sorted_ids])
    wave_padded, wave_lengths = _pad([batch[index][1] for index in sorted_ids])
    phone_padded, phone_lengths = _pad([batch[index][2] for index in sorted_ids], phone=True)
    sid = np.array([int(batch[index][-1][0]) for index in sorted_ids], dtype=np.int64)
    if not use_f0:
        return phone_padded, phone_lengths, spec_padded, spec_lengths, wave_padded, wave_lengths, sid
    pitch_padded = np.zeros((len(batch), phone_padded.shape[1]), dtype=np.int64)
    pitchf_padded = np.zeros((len(batch), phone_padded.shape[1]), dtype=np.float32)
    for out_index, batch_index in enumerate(sorted_ids):
        pitch = batch[batch_index][3]
        pitchf = batch[batch_index][4]
        pitch_padded[out_index, : pitch.shape[0]] = pitch
        pitchf_padded[out_index, : pitchf.shape[0]] = pitchf
    return phone_padded, phone_lengths, pitch_padded, pitchf_padded, spec_padded, spec_lengths, wave_padded, wave_lengths, sid


def _load_phone(path: Path) -> np.ndarray:
    phone = np.load(path, allow_pickle=False)
    return np.repeat(phone, 2, axis=0)[:900].astype(np.float32, copy=False)


def _trim_f0(spec, wav, phone, pitch, pitchf, hop_length: int):
    length = min(phone.shape[0], spec.shape[1], pitch.shape[0], pitchf.shape[0])
    return spec[:, :length], wav[:, : length * hop_length], phone[:length], pitch[:length], pitchf[:length]


def _trim_nof0(spec, wav, phone, hop_length: int):
    length = min(phone.shape[0], spec.shape[1])
    return spec[:, :length], wav[:, : length * hop_length], phone[:length]


def _pad(items: list[np.ndarray], phone: bool = False) -> tuple[np.ndarray, np.ndarray]:
    time_dim = 0 if phone else 1
    max_len = max(item.shape[time_dim] for item in items)
    shape = (len(items), max_len, items[0].shape[1]) if phone else (len(items), items[0].shape[0], max_len)
    out = np.zeros(shape, dtype=np.float32)
    lengths = np.zeros((len(items),), dtype=np.int64)
    for index, item in enumerate(items):
        length = item.shape[time_dim]
        if phone:
            out[index, :length, :] = item
        else:
            out[index, :, :length] = item
        lengths[index] = length
    return out, lengths


def _create_buckets(lengths: list[int], boundaries: list[int]) -> list[list[int]]:
    buckets = [[] for _ in range(len(boundaries) - 1)]
    for index, length in enumerate(lengths):
        bucket_index = next((i for i in range(len(boundaries) - 1) if boundaries[i] < length <= boundaries[i + 1]), -1)
        if bucket_index >= 0:
            buckets[bucket_index].append(index)
    return [bucket for bucket in buckets if bucket]

