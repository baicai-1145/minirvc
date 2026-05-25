from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from minirvc.content.hubert import load_hubert
from minirvc.f0.extract_f0 import F0Coarse
from minirvc.f0.rmvpe import RMVPE
from minirvc.index.search import FeatureIndex
from minirvc.infer.checkpoint import LoadedVoice, load_voice_model
from minirvc.preprocess.audio_io import load_audio, write_wav


class RvcInferencer:
    def __init__(
        self,
        voice_path: str | Path,
        hubert_path: str | Path,
        rmvpe_path: str | Path | None = None,
        index_path: str | Path | None = None,
        index_rate: float = 0.0,
        index_top_k: int = 8,
        index_query_chunk_size: int = 1024,
        device: str | torch.device | None = None,
        half: bool = True,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.voice: LoadedVoice = load_voice_model(voice_path, self.device, half=half)
        self.hubert = load_hubert(hubert_path, device=self.device)
        if self.voice.use_f0 and rmvpe_path is None:
            raise ValueError("rmvpe_path is required for f0 voice models")
        self.rmvpe = RMVPE(rmvpe_path, device=self.device) if self.voice.use_f0 else None
        self.coarse = F0Coarse()
        self.index_rate = float(index_rate)
        self.index_top_k = int(index_top_k)
        self.index_query_chunk_size = int(index_query_chunk_size)
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("index_rate must be between 0 and 1")
        if self.index_top_k <= 0:
            raise ValueError("index_top_k must be positive")
        if self.index_query_chunk_size <= 0:
            raise ValueError("index_query_chunk_size must be positive")
        if self.index_rate > 0 and index_path is None:
            raise ValueError("index_path is required when index_rate > 0")
        self.index = FeatureIndex(index_path, self.device, dtype=torch.float32) if self.index_rate > 0 else None

    def infer_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sid: int = 0,
        f0_up_key: float = 0.0,
        f0_threshold: float = 0.03,
        noise_scale: float = 0.66666,
        protect: float = 0.33,
        highpass_hz: float = 48.0,
        max_peak: float = 0.99,
        split_pad_seconds: int | None = None,
        split_query_seconds: int | None = None,
        split_center_seconds: int | None = None,
        split_max_seconds: int | None = None,
    ) -> np.ndarray:
        audio = load_audio(input_path, 16000, highpass_hz=highpass_hz)
        converted = self.infer_audio(
            audio,
            sid,
            f0_up_key,
            f0_threshold,
            noise_scale,
            protect,
            max_peak,
            split_pad_seconds,
            split_query_seconds,
            split_center_seconds,
            split_max_seconds,
        )
        write_wav(output_path, converted, self.voice.sample_rate)
        return converted

    @torch.inference_mode()
    def infer_audio(
        self,
        audio_16k: np.ndarray,
        sid: int = 0,
        f0_up_key: float = 0.0,
        f0_threshold: float = 0.03,
        noise_scale: float = 0.66666,
        protect: float = 0.33,
        max_peak: float = 0.99,
        split_pad_seconds: int | None = None,
        split_query_seconds: int | None = None,
        split_center_seconds: int | None = None,
        split_max_seconds: int | None = None,
    ) -> np.ndarray:
        audio_16k = np.asarray(audio_16k, dtype=np.float32)
        pad_seconds, query_seconds, center_seconds, max_seconds = self._split_seconds(
            split_pad_seconds,
            split_query_seconds,
            split_center_seconds,
            split_max_seconds,
        )
        window = 160
        pad = 16000 * pad_seconds
        target_pad = self.voice.sample_rate * pad_seconds
        padded = np.pad(audio_16k, (pad, pad), mode="reflect")
        f0 = self._f0(padded, f0_up_key, f0_threshold) if self.voice.use_f0 else None
        if f0 is not None:
            f0 = f0[: padded.shape[0] // window].astype(np.float32, copy=False)

        sid_tensor = torch.tensor([sid], dtype=torch.long, device=self.device)
        pieces: list[np.ndarray] = []
        start = 0
        last_cut: int | None = None
        for cut in split_points(audio_16k, query_seconds, center_seconds, max_seconds):
            cut = cut // window * window
            pieces.append(
                self._infer_padded_chunk(
                    padded[start : cut + 2 * pad + window],
                    sid_tensor,
                    None if f0 is None else f0[start // window : (cut + 2 * pad) // window],
                    noise_scale,
                    protect,
                )[target_pad:-target_pad]
            )
            start = cut
            last_cut = cut

        tail = padded if last_cut is None else padded[last_cut:]
        tail_f0 = None
        if f0 is not None:
            tail_f0 = f0 if last_cut is None else f0[last_cut // window :]
        pieces.append(
            self._infer_padded_chunk(tail, sid_tensor, tail_f0, noise_scale, protect)[target_pad:-target_pad]
        )
        output = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        peak = float(np.abs(output).max()) if output.size else 0.0
        if peak > max_peak:
            output = output / peak * max_peak
        return output.astype(np.float32, copy=False)

    def _infer_padded_chunk(
        self,
        audio_16k: np.ndarray,
        sid_tensor: torch.Tensor,
        f0: np.ndarray | None,
        noise_scale: float,
        protect: float,
    ) -> np.ndarray:
        features = self._content_features(audio_16k, f0, protect)
        frame_count = min(features.shape[1], max(1, audio_16k.shape[0] // 160))
        if f0 is not None:
            frame_count = min(frame_count, f0.shape[0])
        if frame_count <= 0:
            raise ValueError("input audio is too short for inference")

        dtype = next(self.voice.model.parameters()).dtype
        phone = features[:, :frame_count].to(dtype=dtype)
        lengths = torch.tensor([frame_count], dtype=torch.long, device=self.device)
        if f0 is None:
            audio = self.voice.model.infer(phone, lengths, sid_tensor, noise_scale=noise_scale)
        else:
            f0 = f0[:frame_count].astype(np.float32, copy=False)
            pitch = torch.from_numpy(self.coarse(f0)).long().unsqueeze(0).to(self.device)
            pitchf = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)
            audio = self.voice.model.infer(phone, lengths, sid_tensor, pitch, pitchf, noise_scale=noise_scale)

        return audio[0, 0].float().cpu().numpy()

    def _content_features(
        self,
        audio_16k: np.ndarray,
        f0: np.ndarray | None = None,
        protect: float = 0.33,
    ) -> torch.Tensor:
        source = torch.from_numpy(audio_16k).float().view(1, -1).to(self.device)
        features = self.hubert.infer(source, version=self.voice.version)
        original = features
        if self.index is not None:
            features = self.index.blend(features, self.index_rate, self.index_top_k, self.index_query_chunk_size)
        features = F.interpolate(features.transpose(1, 2), scale_factor=2, mode="nearest").transpose(1, 2)
        if f0 is None or protect >= 0.5:
            return features
        original = F.interpolate(original.transpose(1, 2), scale_factor=2, mode="nearest").transpose(1, 2)
        frame_count = min(features.shape[1], original.shape[1], f0.shape[0])
        mask = torch.from_numpy(f0[:frame_count]).float().to(self.device)
        mask[mask > 0] = 1
        mask[mask < 1] = float(protect)
        mask = mask.view(1, frame_count, 1).to(features.dtype)
        mixed = features[:, :frame_count] * mask + original[:, :frame_count] * (1 - mask)
        if frame_count == features.shape[1]:
            return mixed.to(original.dtype)
        return torch.cat([mixed, features[:, frame_count:]], dim=1).to(original.dtype)

    def _f0(self, audio_16k: np.ndarray, f0_up_key: float, threshold: float) -> np.ndarray:
        assert self.rmvpe is not None
        f0 = self.rmvpe.infer_from_audio(audio_16k, threshold=threshold).astype(np.float32, copy=False)
        if f0_up_key:
            f0 *= float(2 ** (f0_up_key / 12.0))
        return f0

    def _split_seconds(
        self,
        pad_seconds: int | None,
        query_seconds: int | None,
        center_seconds: int | None,
        max_seconds: int | None,
    ) -> tuple[int, int, int, int]:
        if (
            pad_seconds is not None
            and query_seconds is not None
            and center_seconds is not None
            and max_seconds is not None
        ):
            resolved = (pad_seconds, query_seconds, center_seconds, max_seconds)
            if min(resolved) <= 0 or query_seconds > center_seconds or center_seconds > max_seconds:
                raise ValueError("split seconds must be positive and satisfy query <= center <= max")
            return resolved
        defaults = (3, 10, 60, 65) if self.voice.use_half else (1, 6, 38, 41)
        values = (pad_seconds, query_seconds, center_seconds, max_seconds)
        resolved = tuple(default if value is None else value for value, default in zip(values, defaults))
        if min(resolved) <= 0 or resolved[1] > resolved[2] or resolved[2] > resolved[3]:
            raise ValueError("split seconds must be positive and satisfy query <= center <= max")
        return resolved[0], resolved[1], resolved[2], resolved[3]


def split_points(
    audio_16k: np.ndarray,
    query_seconds: int,
    center_seconds: int,
    max_seconds: int,
    sample_rate: int = 16000,
    window: int = 160,
) -> list[int]:
    audio = np.asarray(audio_16k)
    audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
    if audio_pad.shape[0] <= sample_rate * max_seconds:
        return []
    audio_sum = np.zeros_like(audio)
    for offset in range(window):
        audio_sum += np.abs(audio_pad[offset : offset - window])
    query = sample_rate * query_seconds
    center = sample_rate * center_seconds
    return [
        t - query + int(np.argmin(audio_sum[t - query : t + query]))
        for t in range(center, audio.shape[0], center)
    ]
