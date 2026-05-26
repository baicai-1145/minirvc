from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from minirvc.f0.extract_f0 import F0Coarse
from minirvc.mlx.device import resolve_device
from minirvc.mlx.weights import (
    as_numpy,
    conv2d_weight,
    conv_transpose2d_weight,
    float_array,
    load_npz_weights,
    load_torch_checkpoint,
    validate_weight_keys,
)
from minirvc.preprocess.audio_io import load_audio_unfiltered


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        if num_layers != 1:
            raise ValueError("MLX RMVPE currently supports the single-layer BiGRU used by RMVPE")
        self.fw = nn.GRU(input_features, hidden_features)
        self.bw = nn.GRU(input_features, hidden_features)

    def __call__(self, x):
        forward = self.fw(x)
        backward = self.bw(x[:, ::-1, :])[:, ::-1, :]
        return mx.concatenate([forward, backward], axis=-1)


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def __call__(self, x):
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
        self.conv = [ConvBlockRes(in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

    def __call__(self, x):
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
        self.bn = nn.BatchNorm(in_channels, momentum=momentum)
        self.layers = []
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def __call__(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int):
        super().__init__()
        self.layers = [ResEncoderBlock(in_channels, out_channels, None, n_blocks)]
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks))

    def __call__(self, x):
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
            nn.BatchNorm(out_channels, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = [ConvBlockRes(out_channels * 2, out_channels)]
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels))

    def __call__(self, x, concat_tensor):
        x = self.conv1(x)
        x = mx.concatenate((x, concat_tensor), axis=-1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels: int, n_decoders: int, stride: tuple[int, int], n_blocks: int):
        super().__init__()
        self.layers = []
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks))
            in_channels = out_channels

    def __call__(self, x, concat_tensors):
        for index, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - index])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size: tuple[int, int], n_blocks: int):
        super().__init__()
        self.encoder = Encoder(1, 128, 5, kernel_size, n_blocks, 16)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, 4, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, 5, kernel_size, n_blocks)

    def __call__(self, x):
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

    def __call__(self, mel):
        x = mx.expand_dims(mel.transpose(0, 2, 1), axis=-1)
        x = self.cnn(self.unet(x))
        batch, time, freq, channels = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(batch, time, channels * freq)
        return self.fc(x)


class MelSpectrogram:
    def __init__(self):
        mel_basis = mel_filter_bank(16000, 1024, 128, 30, 8000, htk=True)
        self.mel_basis = mx.array(mel_basis, dtype=mx.float32)
        self.window = mx.array(np.hanning(1025)[:-1].astype(np.float32), dtype=mx.float32)

    def __call__(self, audio):
        frames = _stft_frames(audio, n_fft=1024, hop_length=160)
        fft = mx.fft.rfft(frames * self.window, n=1024, axis=-1)
        magnitude = mx.abs(fft)
        mel_output = self.mel_basis @ magnitude.swapaxes(1, 2)
        return mx.log(mx.maximum(mel_output, 1e-5))


class RMVPE:
    def __init__(self, model_path: str | Path, device: str | None = None):
        resolve_device(device)
        self.mel_extractor = MelSpectrogram()
        self.model = E2E()
        weights = load_npz_weights(model_path) if str(model_path).endswith(".npz") else _load_torch_rmvpe_weights(model_path)
        validate_weight_keys(self.model, weights, "RMVPE")
        self.model.load_weights([(key, mx.array(value)) for key, value in weights], strict=False)
        self.model.eval()
        mx.eval(self.model.parameters())
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def infer_from_audio(self, audio: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        tensor = mx.array(np.asarray(audio, dtype=np.float32), dtype=mx.float32).reshape(1, -1)
        mel = self.mel_extractor(tensor)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = mx.pad(mel, [(0, 0), (0, 0), (0, n_pad)])
        hidden = self.model(mel.astype(mx.float32))[:, :n_frames]
        mx.eval(hidden)
        return self.decode(np.array(hidden[0], dtype=np.float32), threshold)

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


def extract_f0_directory_mlx(
    exp_dir: str | Path,
    model_path: str | Path,
    workers: int,
    device: str | None,
    batch_size: int = 8,
) -> None:
    import multiprocessing

    exp_dir = Path(exp_dir)
    wav_dir = exp_dir / "1_16k_wavs"
    coarse_dir = exp_dir / "2a_f0"
    nsf_dir = exp_dir / "2b-f0nsf"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    nsf_dir.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(wav_dir.iterdir()) if path.suffix == ".wav" and "spec" not in str(path)]
    worker_count = max(1, workers)
    tasks = [(files[index::worker_count], exp_dir, Path(model_path), device, batch_size) for index in range(worker_count)]
    if workers <= 1:
        _extract_part(tasks[0])
        return
    with multiprocessing.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_extract_part, tasks, chunksize=1):
            pass


def _extract_part(task: tuple[list[Path], Path, Path, str | None, int]) -> None:
    files, exp_dir, model_path, device, batch_size = task
    model = RMVPE(model_path, device=device)
    coarse = F0Coarse()
    if batch_size > 1:
        items = [(path, load_audio_unfiltered(path, 16000)) for path in files]
        for group in _keyed_batches(items, batch_size, lambda item: _rmvpe_padded_frames(item[1].shape[0])):
            _extract_f0_batch(group, model, coarse, exp_dir)
        return
    for path in files:
        f0 = model.infer_from_audio(load_audio_unfiltered(path, 16000), threshold=0.03)
        np.save(exp_dir / "2b-f0nsf" / path.name, f0, allow_pickle=False)
        np.save(exp_dir / "2a_f0" / path.name, coarse(f0), allow_pickle=False)


def _extract_f0_batch(group: list[tuple[Path, np.ndarray]], model: RMVPE, coarse: F0Coarse, exp_dir: Path) -> None:
    mels = []
    frame_lengths = []
    for _, audio in group:
        mel = model.mel_extractor(mx.array(audio, dtype=mx.float32).reshape(1, -1))[0]
        mels.append(mel)
        frame_lengths.append(mel.shape[-1])
    padded_frames = _next_multiple(max(frame_lengths), 32)
    batch_np = np.zeros((len(group), 128, padded_frames), dtype=np.float32)
    for index, mel in enumerate(mels):
        batch_np[index, :, : mel.shape[-1]] = np.array(mel, dtype=np.float32)
    batch = mx.array(batch_np, dtype=mx.float32)
    hidden = model.model(batch)[:, : max(frame_lengths)]
    mx.eval(hidden)
    hidden_np = np.array(hidden, dtype=np.float32)
    for index, (path, _) in enumerate(group):
        f0 = model.decode(hidden_np[index, : frame_lengths[index]], threshold=0.03)
        np.save(exp_dir / "2b-f0nsf" / path.name, f0, allow_pickle=False)
        np.save(exp_dir / "2a_f0" / path.name, coarse(f0), allow_pickle=False)


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


def _stft_frames(audio, n_fft: int, hop_length: int):
    audio_np = np.array(audio, dtype=np.float32)
    padded = np.pad(audio_np, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    x = mx.array(padded, dtype=mx.float32)
    batch, samples = x.shape
    frame_count = 1 + (samples - n_fft) // hop_length
    return mx.as_strided(x, shape=(batch, frame_count, n_fft), strides=(samples, hop_length, 1))


def _load_torch_rmvpe_weights(path: str | Path) -> list[tuple[str, np.ndarray]]:
    state = load_torch_checkpoint(path)
    weights = []
    weights.extend(_gru_weights(state, "fc.0.gru", "fc.layers.0", 256))
    for key, raw_value in state.items():
        if key.startswith("fc.0.gru.") or key.endswith(".num_batches_tracked"):
            continue
        value = as_numpy(raw_value)
        target = _map_rmvpe_key(key)
        if key.endswith(".weight") and value.ndim == 4:
            value = conv_transpose2d_weight(value) if _is_conv_transpose2d_key(key) else conv2d_weight(value)
        elif value.dtype.kind == "f":
            value = float_array(value)
        weights.append((target, value))
    return weights


def _gru_weights(state, source_prefix: str, target_prefix: str, hidden_size: int) -> list[tuple[str, np.ndarray]]:
    out = []
    for suffix, target in (("", "fw"), ("_reverse", "bw")):
        weight_ih = as_numpy(state[f"{source_prefix}.weight_ih_l0{suffix}"])
        weight_hh = as_numpy(state[f"{source_prefix}.weight_hh_l0{suffix}"])
        bias_ih = as_numpy(state[f"{source_prefix}.bias_ih_l0{suffix}"])
        bias_hh = as_numpy(state[f"{source_prefix}.bias_hh_l0{suffix}"])
        bias = np.concatenate(
            [
                bias_ih[: 2 * hidden_size] + bias_hh[: 2 * hidden_size],
                bias_ih[2 * hidden_size :],
            ]
        ).astype(np.float32)
        bhn = bias_hh[2 * hidden_size :].astype(np.float32)
        out.extend(
            [
                (f"{target_prefix}.{target}.Wx", weight_ih.astype(np.float32)),
                (f"{target_prefix}.{target}.Wh", weight_hh.astype(np.float32)),
                (f"{target_prefix}.{target}.b", bias),
                (f"{target_prefix}.{target}.bhn", bhn),
            ]
        )
    return out


def _map_rmvpe_key(key: str) -> str:
    key = re.sub(r"\.conv\.(\d+)\.conv\.(\d+)\.", r".conv.\1.conv.layers.\2.", key)
    key = re.sub(r"\.conv2\.(\d+)\.conv\.(\d+)\.", r".conv2.\1.conv.layers.\2.", key)
    key = re.sub(r"\.conv1\.(\d+)\.", r".conv1.layers.\1.", key)
    key = re.sub(r"^fc\.(\d+)\.", r"fc.layers.\1.", key)
    return key


def _is_conv_transpose2d_key(key: str) -> bool:
    return re.search(r"\.conv1\.0\.weight$", key) is not None


def _keyed_batches(items: list[tuple[Path, np.ndarray]], batch_size: int, key_fn) -> list[list[tuple[Path, np.ndarray]]]:
    buckets: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    groups: list[list[tuple[Path, np.ndarray]]] = []
    for _, bucket in sorted(buckets.items(), key=lambda pair: pair[0], reverse=True):
        groups.extend(bucket[index : index + batch_size] for index in range(0, len(bucket), batch_size))
    return groups


def _next_multiple(value: int, multiple: int) -> int:
    return multiple * ((value - 1) // multiple + 1)


def _rmvpe_padded_frames(samples: int) -> int:
    return _next_multiple(samples // 160 + 1, 32)
