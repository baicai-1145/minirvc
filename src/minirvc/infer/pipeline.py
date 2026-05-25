from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from minirvc.content.hubert import load_hubert
from minirvc.f0.extract_f0 import F0Coarse
from minirvc.f0.rmvpe import RMVPE
from minirvc.infer.checkpoint import LoadedVoice, load_voice_model
from minirvc.preprocess.audio_io import load_audio, write_wav


class RvcInferencer:
    def __init__(
        self,
        voice_path: str | Path,
        hubert_path: str | Path,
        rmvpe_path: str | Path | None = None,
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

    def infer_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sid: int = 0,
        f0_up_key: float = 0.0,
        f0_threshold: float = 0.03,
        noise_scale: float = 0.66666,
        highpass_hz: float = 48.0,
        max_peak: float = 0.99,
    ) -> np.ndarray:
        audio = load_audio(input_path, 16000, highpass_hz=highpass_hz)
        converted = self.infer_audio(audio, sid, f0_up_key, f0_threshold, noise_scale, max_peak)
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
        max_peak: float = 0.99,
    ) -> np.ndarray:
        audio_16k = np.asarray(audio_16k, dtype=np.float32)
        features = self._content_features(audio_16k)
        f0 = self._f0(audio_16k, f0_up_key, f0_threshold) if self.voice.use_f0 else None
        frame_count = min(features.shape[1], max(1, audio_16k.shape[0] // 160))
        if f0 is not None:
            frame_count = min(frame_count, f0.shape[0])
        if frame_count <= 0:
            raise ValueError("input audio is too short for inference")

        dtype = next(self.voice.model.parameters()).dtype
        phone = features[:, :frame_count].to(dtype=dtype)
        lengths = torch.tensor([frame_count], dtype=torch.long, device=self.device)
        sid_tensor = torch.tensor([sid], dtype=torch.long, device=self.device)
        if f0 is None:
            audio = self.voice.model.infer(phone, lengths, sid_tensor, noise_scale=noise_scale)
        else:
            f0 = f0[:frame_count].astype(np.float32, copy=False)
            pitch = torch.from_numpy(self.coarse(f0)).long().unsqueeze(0).to(self.device)
            pitchf = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)
            audio = self.voice.model.infer(phone, lengths, sid_tensor, pitch, pitchf, noise_scale=noise_scale)

        output = audio[0, 0].float().cpu().numpy()
        peak = float(np.abs(output).max()) if output.size else 0.0
        if peak > max_peak:
            output = output / peak * max_peak
        return output.astype(np.float32, copy=False)

    def _content_features(self, audio_16k: np.ndarray) -> torch.Tensor:
        source = torch.from_numpy(audio_16k).float().view(1, -1).to(self.device)
        features = self.hubert.infer(source, version=self.voice.version)
        return F.interpolate(features.transpose(1, 2), scale_factor=2, mode="nearest").transpose(1, 2)

    def _f0(self, audio_16k: np.ndarray, f0_up_key: float, threshold: float) -> np.ndarray:
        assert self.rmvpe is not None
        f0 = self.rmvpe.infer_from_audio(audio_16k, threshold=threshold).astype(np.float32, copy=False)
        if f0_up_key:
            f0 *= float(2 ** (f0_up_key / 12.0))
        return f0
