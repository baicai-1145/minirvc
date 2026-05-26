from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from minirvc.f0.extract_f0 import F0Coarse
from minirvc.mlx.device import resolve_device
from minirvc.mlx.hubert import load_hubert
from minirvc.mlx.rmvpe import RMVPE
from minirvc.mlx.train_models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from minirvc.preprocess.audio_io import load_audio, write_wav


@dataclass(frozen=True)
class LoadedMlxVoice:
    model: object
    sample_rate: int
    use_f0: bool
    version: str
    precision: str


class MlxRvcInferencer:
    def __init__(
        self,
        voice_path: str | Path,
        hubert_path: str | Path,
        rmvpe_path: str | Path | None = None,
        index_path: str | Path | None = None,
        index_rate: float = 0.0,
        index_top_k: int = 8,
        index_query_chunk_size: int = 1024,
        device: str | None = None,
        precision: str = "fp32",
    ):
        resolve_device(device)
        self.dtype = _resolve_precision(precision)
        self.low_precision = self.dtype != mx.float32
        self.voice = load_voice_model_mlx(voice_path, precision=precision)
        self.hubert = load_hubert(hubert_path, device=device)
        if self.voice.use_f0 and rmvpe_path is None:
            raise ValueError("rmvpe_path is required for f0 voice models")
        self.rmvpe = RMVPE(rmvpe_path, device=device) if self.voice.use_f0 else None
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
        self.index = MlxFeatureIndex(index_path) if self.index_rate > 0 else None

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

        sid_tensor = mx.array([sid], dtype=mx.int32)
        pieces: list[np.ndarray] = []
        start = 0
        last_cut: int | None = None
        for cut in split_points(audio_16k, query_seconds, center_seconds, max_seconds):
            cut = cut // window * window
            chunk = self._infer_padded_chunk(
                padded[start : cut + 2 * pad + window],
                sid_tensor,
                None if f0 is None else f0[start // window : (cut + 2 * pad) // window],
                noise_scale,
                protect,
            )
            pieces.append(chunk[target_pad:-target_pad])
            start = cut
            last_cut = cut

        tail = padded if last_cut is None else padded[last_cut:]
        tail_f0 = None if f0 is None else (f0 if last_cut is None else f0[last_cut // window :])
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
        sid_tensor,
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

        phone = features[:, :frame_count].astype(self.dtype)
        lengths = mx.array([frame_count], dtype=mx.int32)
        if f0 is None:
            audio = self.voice.model.infer(phone, lengths, sid_tensor, noise_scale=noise_scale)
        else:
            f0 = f0[:frame_count].astype(np.float32, copy=False)
            pitch = mx.array(self.coarse(f0), dtype=mx.int32)[None, :]
            pitchf = mx.array(f0, dtype=self.dtype)[None, :]
            audio = self.voice.model.infer(phone, lengths, sid_tensor, pitch, pitchf, noise_scale=noise_scale)
        mx.eval(audio)
        return np.array(audio[0, 0].astype(mx.float32), dtype=np.float32)

    def _content_features(
        self,
        audio_16k: np.ndarray,
        f0: np.ndarray | None = None,
        protect: float = 0.33,
    ):
        source = mx.array(np.asarray(audio_16k, dtype=np.float32), dtype=mx.float32).reshape(1, -1)
        features = self.hubert.infer(source, version=self.voice.version)
        original = features
        if self.index is not None:
            features = self.index.blend(features, self.index_rate, self.index_top_k, self.index_query_chunk_size)
        features = mx.repeat(features, repeats=2, axis=1)
        if f0 is None or protect >= 0.5:
            return features
        original = mx.repeat(original, repeats=2, axis=1)
        frame_count = min(features.shape[1], original.shape[1], f0.shape[0])
        mask = mx.array(f0[:frame_count], dtype=mx.float32)
        mask = mx.where(mask > 0, 1.0, mask)
        mask = mx.where(mask < 1, float(protect), mask).reshape(1, frame_count, 1).astype(features.dtype)
        mixed = features[:, :frame_count] * mask + original[:, :frame_count] * (1 - mask)
        if frame_count == features.shape[1]:
            return mixed.astype(original.dtype)
        return mx.concatenate([mixed, features[:, frame_count:]], axis=1).astype(original.dtype)

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
        defaults = (3, 10, 60, 65) if self.low_precision else (1, 6, 38, 41)
        values = (pad_seconds, query_seconds, center_seconds, max_seconds)
        resolved = tuple(default if value is None else value for value, default in zip(values, defaults))
        if min(resolved) <= 0 or resolved[1] > resolved[2] or resolved[2] > resolved[3]:
            raise ValueError("split seconds must be positive and satisfy query <= center <= max")
        return resolved[0], resolved[1], resolved[2], resolved[3]


class MlxFeatureIndex:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"index features must be 2D, got shape={features.shape}")
        self.features = mx.array(features, dtype=mx.float32)
        self.search_features_t = self.features.T
        self.squared_norms = mx.sum(self.features * self.features, axis=1)
        mx.eval(self.features, self.search_features_t, self.squared_norms)

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def blend(self, query, rate: float, top_k: int = 8, query_chunk_size: int = 1024):
        if rate <= 0:
            return query
        if query.shape[-1] != self.dim:
            raise ValueError(f"index dim {self.dim} does not match query dim {query.shape[-1]}")
        flat_query = query.astype(mx.float32)[0]
        k = min(int(top_k), int(self.features.shape[0]))
        chunks = []
        for start in range(0, flat_query.shape[0], query_chunk_size):
            chunk = self._retrieve_chunk(flat_query[start : start + query_chunk_size], k)
            mx.eval(chunk)
            chunks.append(chunk)
        retrieved = mx.concatenate(chunks, axis=0)[None, :, :]
        mixed = retrieved.astype(query.dtype) * float(rate) + query * float(1.0 - rate)
        return mixed.astype(query.dtype)

    def _squared_l2(self, query) -> mx.array:
        query_norms = mx.sum(query * query, axis=1, keepdims=True)
        distances = query_norms + self.squared_norms[None, :] - 2.0 * (query @ self.search_features_t)
        distances = mx.nan_to_num(distances, nan=float("inf"), posinf=float("inf"), neginf=0.0)
        return mx.maximum(distances, 0.0)

    def _retrieve_chunk(self, query, k: int):
        distances = self._squared_l2(query)
        indices = mx.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        values = mx.take_along_axis(distances, indices, axis=1)
        weights = mx.square(1.0 / mx.maximum(values, 1e-12))
        weights = weights / mx.sum(weights, axis=1, keepdims=True)
        return mx.sum(self.features[indices] * weights[:, :, None], axis=1)


def load_voice_model_mlx(path: str | Path, precision: str = "fp32") -> LoadedMlxVoice:
    if not str(path).endswith(".npz"):
        raise ValueError("MLX inference expects a .mlx.npz voice model")
    dtype = _resolve_precision(precision)
    with np.load(path, allow_pickle=True) as data:
        config = list(data["__config__"])
        version = str(np.asarray(data["__version__"]).item())
        use_f0 = bool(int(np.asarray(data["__f0__"]).reshape(-1)[0]))
        sample_rate = int(config[-1])
        cls = {
            ("v1", True): SynthesizerTrnMs256NSFsid,
            ("v1", False): SynthesizerTrnMs256NSFsid_nono,
            ("v2", True): SynthesizerTrnMs768NSFsid,
            ("v2", False): SynthesizerTrnMs768NSFsid_nono,
        }[(version, use_f0)]
        model = cls(*config, is_half=False)
        expected = {key for key, _ in tree_flatten(model.parameters())}
        weights = []
        for key in data.files:
            if key.startswith("__"):
                continue
            if key not in expected:
                raise RuntimeError(f"unexpected MLX voice checkpoint key: {key}")
            weights.append((key, mx.array(np.asarray(data[key]))))
    loaded = {key for key, _ in weights}
    missing = sorted(key for key in expected - loaded if not key.startswith("enc_q."))
    if missing:
        raise RuntimeError(f"MLX voice checkpoint mismatch: missing={missing}")
    model.load_weights(weights, strict=False)
    model.eval()
    if dtype != mx.float32:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return LoadedMlxVoice(model=model, sample_rate=sample_rate, use_f0=use_f0, version=version, precision=precision)


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


def _resolve_precision(precision: str):
    if precision == "fp32":
        return mx.float32
    if precision == "bf16":
        return mx.bfloat16
    if precision == "fp16":
        return mx.float16
    raise ValueError("precision must be one of: fp32, bf16, fp16")
