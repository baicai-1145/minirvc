from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from minirvc.preprocess.audio_io import load_audio_unfiltered
from minirvc.train.mel import spectrogram_torch


@dataclass(frozen=True)
class TrainItem:
    wav_path: Path
    feature_path: Path
    sid: int
    f0_path: Path | None = None
    f0nsf_path: Path | None = None


def load_filelist(path: str | Path, use_f0: bool) -> list[TrainItem]:
    items: list[TrainItem] = []
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


class RvcDataset(Dataset):
    def __init__(self, filelist: str | Path, hps, use_f0: bool):
        self.items = load_filelist(filelist, use_f0)
        self.use_f0 = use_f0
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
            feature_len = _feature_length(item.feature_path)
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
        audio = torch.from_numpy(load_audio_unfiltered(item.wav_path, self.sampling_rate)).float()
        wav = audio.unsqueeze(0)
        spec = spectrogram_torch(
            wav,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).squeeze(0)
        sid = torch.LongTensor([item.sid])

        if self.use_f0:
            assert item.f0_path is not None and item.f0nsf_path is not None
            pitch = torch.LongTensor(np.load(item.f0_path, allow_pickle=False))
            pitchf = torch.FloatTensor(np.load(item.f0nsf_path, allow_pickle=False))
            return _trim_f0(spec, wav, phone, pitch, pitchf, self.hop_length) + (sid,)
        return _trim_nof0(spec, wav, phone, self.hop_length) + (sid,)


class RvcCollate:
    def __init__(self, use_f0: bool):
        self.use_f0 = use_f0

    def __call__(self, batch):
        ids = torch.argsort(torch.LongTensor([row[0].size(1) for row in batch]), descending=True)
        sorted_ids = [int(index) for index in ids]
        spec_padded, spec_lengths = _pad([batch[index][0] for index in sorted_ids])
        wave_padded, wave_lengths = _pad([batch[index][1] for index in sorted_ids])
        phone_padded, phone_lengths = _pad([batch[index][2] for index in sorted_ids], phone=True)
        sid = torch.LongTensor([int(batch[index][-1]) for index in sorted_ids])
        if not self.use_f0:
            return phone_padded, phone_lengths, spec_padded, spec_lengths, wave_padded, wave_lengths, sid
        pitch_padded = torch.zeros(len(batch), phone_padded.size(1), dtype=torch.long)
        pitchf_padded = torch.zeros(len(batch), phone_padded.size(1), dtype=torch.float32)
        for out_index, batch_index in enumerate(sorted_ids):
            pitch = batch[batch_index][3]
            pitchf = batch[batch_index][4]
            pitch_padded[out_index, : pitch.size(0)] = pitch
            pitchf_padded[out_index, : pitchf.size(0)] = pitchf
        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int, boundaries: list[int], shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets = self._create_buckets()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        batches: list[list[int]] = []
        for bucket in self.buckets:
            order = torch.randperm(len(bucket), generator=generator).tolist() if self.shuffle else list(range(len(bucket)))
            ordered = [bucket[index] for index in order]
            rem = (self.batch_size - (len(ordered) % self.batch_size)) % self.batch_size
            if rem:
                ordered += (ordered * math.ceil(rem / len(ordered)))[:rem]
            for start in range(0, len(ordered), self.batch_size):
                batches.append(ordered[start : start + self.batch_size])
        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(bucket) / self.batch_size) for bucket in self.buckets)

    def _create_buckets(self) -> list[list[int]]:
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for index, length in enumerate(self.lengths):
            bucket_index = self._bucket_index(length)
            if bucket_index >= 0:
                buckets[bucket_index].append(index)
        return [bucket for bucket in buckets if bucket]

    def _bucket_index(self, length: int) -> int:
        return next(
            (index for index in range(len(self.boundaries) - 1) if self.boundaries[index] < length <= self.boundaries[index + 1]),
            -1,
        )


def _feature_length(path: Path) -> int:
    return int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])


def _load_phone(path: Path) -> torch.Tensor:
    phone = np.load(path, allow_pickle=False)
    phone = np.repeat(phone, 2, axis=0)[:900]
    return torch.FloatTensor(phone)


def _trim_f0(
    spec: torch.Tensor,
    wav: torch.Tensor,
    phone: torch.Tensor,
    pitch: torch.Tensor,
    pitchf: torch.Tensor,
    hop_length: int,
):
    length = min(phone.size(0), spec.size(1), pitch.size(0), pitchf.size(0))
    return spec[:, :length], wav[:, : length * hop_length], phone[:length], pitch[:length], pitchf[:length]


def _trim_nof0(spec: torch.Tensor, wav: torch.Tensor, phone: torch.Tensor, hop_length: int):
    length = min(phone.size(0), spec.size(1))
    return spec[:, :length], wav[:, : length * hop_length], phone[:length]


def _pad(items: list[torch.Tensor], phone: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    time_dim = 0 if phone else 1
    max_len = max(item.size(time_dim) for item in items)
    shape = (len(items), max_len, items[0].size(1)) if phone else (len(items), items[0].size(0), max_len)
    out = torch.zeros(*shape, dtype=torch.float32)
    lengths = torch.LongTensor(len(items))
    for index, item in enumerate(items):
        length = item.size(time_dim)
        if phone:
            out[index, :length, :] = item
        else:
            out[index, :, :length] = item
        lengths[index] = length
    return out, lengths
