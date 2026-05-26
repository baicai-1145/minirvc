from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from minirvc.infer.pipeline import RvcInferencer
from minirvc.preprocess.audio_io import resample_audio


@dataclass
class RealtimeConfig:
    voice_path: Path
    hubert_path: Path = Path("assets/hubert/hubert_base.pt")
    rmvpe_path: Path = Path("assets/rmvpe/rmvpe.pt")
    index_path: Path | None = None
    index_rate: float = 0.0
    index_top_k: int = 8
    index_query_chunk_size: int = 1024
    backend: str = "torch"
    device: str | None = None
    half: bool = True
    precision: str = "bf16"
    sid: int = 0
    f0_up_key: float = 0.0
    f0_threshold: float = 0.03
    noise_scale: float = 0.66666
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    sola_search_time: float = 0.01


@dataclass
class RealtimeStats:
    feature_s: float = 0.0
    index_s: float = 0.0
    f0_s: float = 0.0
    model_s: float = 0.0
    total_s: float = 0.0
    sola_offset: int = 0


class RealtimeEngine:
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.inferencer = RvcInferencer(
            voice_path=config.voice_path,
            hubert_path=config.hubert_path,
            rmvpe_path=config.rmvpe_path,
            index_path=config.index_path,
            index_rate=config.index_rate,
            index_top_k=config.index_top_k,
            index_query_chunk_size=config.index_query_chunk_size,
            device=config.device,
            half=config.half,
        )
        self.device = self.inferencer.device
        self.sample_rate = self.inferencer.voice.sample_rate
        self.zc = self.sample_rate // 100
        self.block_frame = self._frames(config.block_time)
        self.crossfade_frame = self._frames(config.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self._frames(config.sola_search_time)
        self.extra_frame = self._frames(config.extra_time)
        self.input_frame = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.input_wav = np.zeros(self.input_frame, dtype=np.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        self.fade_in = torch.sin(
            0.5
            * np.pi
            * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        ).square()
        self.fade_out = 1.0 - self.fade_in
        self.cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.last_stats = RealtimeStats()
        self._warmup()

    def _frames(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate / self.zc)) * self.zc

    @torch.inference_mode()
    def _warmup(self) -> None:
        input_wav = self.input_wav.copy()
        sola_buffer = self.sola_buffer.clone()
        cache_pitch = self.cache_pitch.clone()
        cache_pitchf = self.cache_pitchf.clone()
        last_stats = self.last_stats
        self.process_block(np.zeros(self.block_frame, dtype=np.float32))
        self.input_wav[:] = input_wav
        self.sola_buffer.copy_(sola_buffer)
        self.cache_pitch.copy_(cache_pitch)
        self.cache_pitchf.copy_(cache_pitchf)
        self.last_stats = last_stats

    @torch.inference_mode()
    def process_block(self, block: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.shape[0] != self.block_frame:
            raise ValueError(f"expected block length {self.block_frame}, got {block.shape[0]}")
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :]
        self.input_wav[-self.block_frame :] = block
        audio_16k = self._input_16k()
        converted = self._convert(audio_16k)
        output, sola_offset = self._sola(converted)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_stats.total_s = time.perf_counter() - start
        self.last_stats.sola_offset = int(sola_offset)
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def _input_16k(self) -> np.ndarray:
        audio = resample_audio(self.input_wav, self.sample_rate, 16000)
        expected = 160 * self.input_frame // self.zc
        if audio.shape[0] > expected:
            audio = audio[:expected]
        elif audio.shape[0] < expected:
            audio = np.pad(audio, (0, expected - audio.shape[0]))
        return audio.astype(np.float32, copy=False)

    def _convert(self, audio_16k: np.ndarray) -> torch.Tensor:
        stats = RealtimeStats()
        sid = torch.tensor([self.config.sid], dtype=torch.long, device=self.device)
        source = torch.from_numpy(audio_16k).float().view(1, -1).to(self.device)

        t0 = time.perf_counter()
        features = self.inferencer.hubert.infer(source, version=self.inferencer.voice.version)
        features = torch.cat((features, features[:, -1:, :]), dim=1)
        t1 = time.perf_counter()

        if self.inferencer.index is not None and self.inferencer.index_rate > 0:
            head = self.skip_head // 2
            mixed = self.inferencer.index.blend(
                features[:, head:],
                self.inferencer.index_rate,
                self.inferencer.index_top_k,
                self.inferencer.index_query_chunk_size,
            )
            features = torch.cat((features[:, :head], mixed), dim=1)
        t2 = time.perf_counter()

        p_len = min(audio_16k.shape[0] // 160, features.shape[1] * 2)
        pitch = pitchf = None
        if self.inferencer.voice.use_f0:
            pitch, pitchf = self._f0(audio_16k, p_len)
        t3 = time.perf_counter()

        features = F.interpolate(features.transpose(1, 2), scale_factor=2, mode="nearest").transpose(1, 2)
        features = features[:, :p_len]
        dtype = next(self.inferencer.voice.model.parameters()).dtype
        lengths = torch.tensor([p_len], dtype=torch.long, device=self.device)
        skip = torch.tensor([self.skip_head], dtype=torch.long, device=self.device)
        ret = torch.tensor([self.return_length], dtype=torch.long, device=self.device)
        if self.inferencer.voice.use_f0:
            audio = self.inferencer.voice.model.infer(
                features.to(dtype),
                lengths,
                sid,
                pitch,
                pitchf,
                noise_scale=self.config.noise_scale,
                skip_head=skip,
                return_length=ret,
                return_length2=ret,
            )
        else:
            audio = self.inferencer.voice.model.infer(
                features.to(dtype),
                lengths,
                sid,
                noise_scale=self.config.noise_scale,
                skip_head=skip,
                return_length=ret,
                return_length2=ret,
            )
        t4 = time.perf_counter()
        stats.feature_s = t1 - t0
        stats.index_s = t2 - t1
        stats.f0_s = t3 - t2
        stats.model_s = t4 - t3
        self.last_stats = stats
        return audio[0, 0].float()

    def _f0(self, audio_16k: np.ndarray, p_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count = self.block_frame_16k + 800
        frame_count = 5120 * ((frame_count - 1) // 5120 + 1) - 160
        f0 = self.inferencer._f0(audio_16k[-frame_count:], self.config.f0_up_key, self.config.f0_threshold)
        coarse = self.inferencer.coarse(f0)
        pitch = torch.from_numpy(coarse).long().to(self.device)
        pitchf = torch.from_numpy(f0.astype(np.float32, copy=False)).float().to(self.device)
        shift = max(1, self.block_frame_16k // 160)
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        usable = min(max(0, pitch.numel() - 4), self.cache_pitch.numel())
        if usable:
            self.cache_pitch[-usable:] = pitch[3 : 3 + usable]
            self.cache_pitchf[-usable:] = pitchf[3 : 3 + usable]
        return self.cache_pitch[None, -p_len:], self.cache_pitchf[None, -p_len:]

    def _sola(self, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
        required = self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        if audio.numel() < required:
            audio = F.pad(audio, (0, required - audio.numel()))
        conv_input = audio[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input.square(),
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
        audio = audio[sola_offset:]
        if audio.numel() < self.block_frame + self.sola_buffer_frame:
            audio = F.pad(audio, (0, self.block_frame + self.sola_buffer_frame - audio.numel()))
        audio = audio.clone()
        audio[: self.sola_buffer_frame] *= self.fade_in
        audio[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = audio[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return audio[: self.block_frame], sola_offset


class MlxRealtimeEngine:
    def __init__(self, config: RealtimeConfig):
        from minirvc.mlx.infer import MlxRvcInferencer

        import mlx.core as mx

        self.mx = mx
        self.config = config
        self.inferencer = MlxRvcInferencer(
            voice_path=config.voice_path,
            hubert_path=config.hubert_path,
            rmvpe_path=config.rmvpe_path,
            index_path=config.index_path,
            index_rate=config.index_rate,
            index_top_k=config.index_top_k,
            index_query_chunk_size=config.index_query_chunk_size,
            device=config.device,
            precision=config.precision,
        )
        self.sample_rate = self.inferencer.voice.sample_rate
        self.zc = self.sample_rate // 100
        self.block_frame = self._frames(config.block_time)
        self.crossfade_frame = self._frames(config.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self._frames(config.sola_search_time)
        self.extra_frame = self._frames(config.extra_time)
        self.input_frame = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.input_wav = np.zeros(self.input_frame, dtype=np.float32)
        self.sola_buffer = np.zeros(self.sola_buffer_frame, dtype=np.float32)
        self.fade_in = np.sin(
            0.5 * np.pi * np.linspace(0.0, 1.0, num=self.sola_buffer_frame, dtype=np.float32)
        ).astype(np.float32)
        self.fade_in *= self.fade_in
        self.fade_out = 1.0 - self.fade_in
        self.cache_pitch = np.zeros(1024, dtype=np.int32)
        self.cache_pitchf = np.zeros(1024, dtype=np.float32)
        self.last_stats = RealtimeStats()
        self._warmup()

    def _frames(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate / self.zc)) * self.zc

    def _warmup(self) -> None:
        input_wav = self.input_wav.copy()
        sola_buffer = self.sola_buffer.copy()
        cache_pitch = self.cache_pitch.copy()
        cache_pitchf = self.cache_pitchf.copy()
        last_stats = self.last_stats
        self.process_block(np.zeros(self.block_frame, dtype=np.float32))
        self.input_wav[:] = input_wav
        self.sola_buffer[:] = sola_buffer
        self.cache_pitch[:] = cache_pitch
        self.cache_pitchf[:] = cache_pitchf
        self.last_stats = last_stats

    def process_block(self, block: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.shape[0] != self.block_frame:
            raise ValueError(f"expected block length {self.block_frame}, got {block.shape[0]}")
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :]
        self.input_wav[-self.block_frame :] = block
        audio_16k = self._input_16k()
        converted = self._convert(audio_16k)
        output, sola_offset = self._sola(converted)
        self.last_stats.total_s = time.perf_counter() - start
        self.last_stats.sola_offset = int(sola_offset)
        return output.astype(np.float32, copy=False)

    def _input_16k(self) -> np.ndarray:
        audio = resample_audio(self.input_wav, self.sample_rate, 16000)
        expected = 160 * self.input_frame // self.zc
        if audio.shape[0] > expected:
            audio = audio[:expected]
        elif audio.shape[0] < expected:
            audio = np.pad(audio, (0, expected - audio.shape[0]))
        return audio.astype(np.float32, copy=False)

    def _convert(self, audio_16k: np.ndarray) -> np.ndarray:
        mx = self.mx
        stats = RealtimeStats()
        sid = mx.array([self.config.sid], dtype=mx.int32)
        source = mx.array(audio_16k, dtype=mx.float32).reshape(1, -1)

        t0 = time.perf_counter()
        features = self.inferencer.hubert.infer(source, version=self.inferencer.voice.version)
        features = mx.concatenate((features, features[:, -1:, :]), axis=1)
        mx.eval(features)
        t1 = time.perf_counter()

        if self.inferencer.index is not None and self.inferencer.index_rate > 0:
            head = self.skip_head // 2
            mixed = self.inferencer.index.blend(
                features[:, head:],
                self.inferencer.index_rate,
                self.inferencer.index_top_k,
                self.inferencer.index_query_chunk_size,
            )
            features = mx.concatenate((features[:, :head], mixed), axis=1)
            mx.eval(features)
        t2 = time.perf_counter()

        p_len = min(audio_16k.shape[0] // 160, features.shape[1] * 2)
        pitch = pitchf = None
        if self.inferencer.voice.use_f0:
            pitch, pitchf = self._f0(audio_16k, p_len)
        t3 = time.perf_counter()

        features = mx.repeat(features, repeats=2, axis=1)[:, :p_len]
        lengths = mx.array([p_len], dtype=mx.int32)
        skip = mx.array([self.skip_head], dtype=mx.int32)
        ret = mx.array([self.return_length], dtype=mx.int32)
        if self.inferencer.voice.use_f0:
            audio = self.inferencer.voice.model.infer(
                features.astype(self.inferencer.dtype),
                lengths,
                sid,
                pitch,
                pitchf,
                noise_scale=self.config.noise_scale,
                skip_head=skip,
                return_length=ret,
                return_length2=ret,
            )
        else:
            audio = self.inferencer.voice.model.infer(
                features.astype(self.inferencer.dtype),
                lengths,
                sid,
                noise_scale=self.config.noise_scale,
                skip_head=skip,
                return_length=ret,
                return_length2=ret,
            )
        mx.eval(audio)
        t4 = time.perf_counter()
        stats.feature_s = t1 - t0
        stats.index_s = t2 - t1
        stats.f0_s = t3 - t2
        stats.model_s = t4 - t3
        self.last_stats = stats
        return np.array(audio[0, 0].astype(mx.float32), dtype=np.float32)

    def _f0(self, audio_16k: np.ndarray, p_len: int):
        mx = self.mx
        frame_count = self.block_frame_16k + 800
        frame_count = 5120 * ((frame_count - 1) // 5120 + 1) - 160
        f0 = self.inferencer._f0(audio_16k[-frame_count:], self.config.f0_up_key, self.config.f0_threshold)
        coarse = self.inferencer.coarse(f0).astype(np.int32, copy=False)
        pitchf = f0.astype(np.float32, copy=False)
        shift = max(1, self.block_frame_16k // 160)
        self.cache_pitch[:-shift] = self.cache_pitch[shift:]
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:]
        usable = min(max(0, coarse.shape[0] - 4), self.cache_pitch.shape[0])
        if usable:
            self.cache_pitch[-usable:] = coarse[3 : 3 + usable]
            self.cache_pitchf[-usable:] = pitchf[3 : 3 + usable]
        pitch = mx.array(self.cache_pitch[None, -p_len:], dtype=mx.int32)
        pitchf_tensor = mx.array(self.cache_pitchf[None, -p_len:], dtype=self.inferencer.dtype)
        return pitch, pitchf_tensor

    def _sola(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        required = self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        if audio.shape[0] < required:
            audio = np.pad(audio, (0, required - audio.shape[0]))
        if self.sola_buffer_frame <= 0:
            return audio[: self.block_frame].copy(), 0
        search = audio[: self.sola_buffer_frame + self.sola_search_frame]
        windows = np.lib.stride_tricks.sliding_window_view(search, self.sola_buffer_frame)
        cor_nom = windows @ self.sola_buffer
        cor_den = np.sqrt(np.sum(windows * windows, axis=1) + 1e-8)
        sola_offset = int(np.argmax(cor_nom / cor_den))
        audio = audio[sola_offset:]
        if audio.shape[0] < self.block_frame + self.sola_buffer_frame:
            audio = np.pad(audio, (0, self.block_frame + self.sola_buffer_frame - audio.shape[0]))
        output = audio[: self.block_frame].copy()
        output[: self.sola_buffer_frame] *= self.fade_in
        output[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = audio[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return output, sola_offset


def create_realtime_engine(config: RealtimeConfig):
    if config.backend == "torch":
        return RealtimeEngine(config)
    if config.backend == "mlx":
        return MlxRealtimeEngine(config)
    raise ValueError(f"unsupported realtime backend: {config.backend}")
