from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "shortcut"):
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | None,
        n_blocks: int,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        for conv in self.conv:
            x = conv(x)
        if self.pool is None:
            return x
        return x, self.pool(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], n_blocks: int):
        super().__init__()
        output_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride: tuple[int, int], n_blocks: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - index])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size: tuple[int, int], n_blocks: int):
        super().__init__()
        self.encoder = Encoder(1, 128, 5, kernel_size, n_blocks, 16)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, 4, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, 5, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, concat_tensors)


class E2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = DeepUnet((2, 2), 4)
        self.cnn = nn.Conv2d(16, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, 1),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        mel_basis = mel_filter_bank(16000, 1024, 128, 30, 8000, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("window", torch.hann_window(1024))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        fft = torch.stft(
            audio,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis.to(audio.device), magnitude)
        return torch.log(torch.clamp(mel_output, min=1e-5))


class RMVPE:
    def __init__(self, model_path: str | Path, device: str | torch.device | None = None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.mel_extractor = MelSpectrogram().to(self.device)
        self.model = E2E()
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.float().eval().to(self.device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    @torch.inference_mode()
    def infer_from_audio(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(tensor)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        hidden = self.model(mel.float())[:, :n_frames].squeeze(0).cpu().numpy()
        return self.decode(hidden, threshold)

    def decode(self, hidden: np.ndarray, threshold: float) -> np.ndarray:
        cents_pred = self.to_local_average_cents(hidden, threshold)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def to_local_average_cents(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        starts = center - 4
        ends = center + 5
        todo_salience = np.array([salience[index, starts[index] : ends[index]] for index in range(salience.shape[0])])
        todo_cents_mapping = np.array([self.cents_mapping[starts[index] : ends[index]] for index in range(salience.shape[0])])
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        divided = product_sum / weight_sum
        divided[np.max(salience, axis=1) <= threshold] = 0
        return divided


def mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    htk: bool,
) -> np.ndarray:
    if not htk:
        raise ValueError("RMVPE expects HTK mel filters")
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    ramps = np.subtract.outer(hz_points, fft_freqs)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / (hz_points[i + 1] - hz_points[i])
        upper = ramps[i + 2] / (hz_points[i + 2] - hz_points[i + 1])
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asanyarray(frequencies) / 700.0)


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asanyarray(mels) / 2595.0) - 1.0)
