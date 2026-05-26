from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from minirvc.mlx.device import resolve_device
from minirvc.mlx.weights import as_numpy, conv1d_weight, float_array, load_npz_weights, load_torch_checkpoint, validate_weight_keys


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        specs = [(1, 512, 10, 5), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 2, 2), (512, 512, 2, 2)]
        self.conv_layers = []
        for index, (in_channels, out_channels, kernel, stride) in enumerate(specs):
            if index == 0:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False),
                        nn.Dropout(0.0),
                        nn.GroupNorm(out_channels, out_channels, pytorch_compatible=True),
                        nn.GELU(),
                    )
                )
            else:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False),
                        nn.Dropout(0.0),
                        nn.GELU(),
                    )
                )

    def __call__(self, source):
        x = source[..., None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class WeightNormConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_g = mx.zeros((1, 1, 128))
        self.weight_v = mx.zeros((768, 48, 128))
        self.bias = mx.zeros((768,))

    def __call__(self, x):
        norm = mx.linalg.norm(self.weight_v, axis=(0, 1), keepdims=True) + 1e-12
        weight = self.weight_v * (self.weight_g / norm)
        return mx.conv1d(x, weight.transpose(0, 2, 1), padding=64, groups=16) + self.bias


class SamePad(nn.Module):
    def __call__(self, x):
        return x[:, :-1, :]


class HubertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.q_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)
        self.num_heads = 12
        self.head_dim = 64
        self.scaling = self.head_dim**-0.5

    def __call__(self, x, padding_mask=None):
        batch, time, embed = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._shape(q, batch, time)
        k = self._shape(k, batch, time)
        v = self._shape(v, batch, time)

        weights = q @ k.swapaxes(1, 2)
        if padding_mask is not None and bool(np.array(mx.any(padding_mask))):
            weights = weights.reshape(batch, self.num_heads, time, time)
            weights = mx.where(padding_mask[:, None, None, :], -np.inf, weights)
            weights = weights.reshape(batch * self.num_heads, time, time)
        probs = mx.softmax(weights.astype(mx.float32), axis=-1)
        attn = probs @ v
        attn = attn.reshape(batch, self.num_heads, time, self.head_dim)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, time, embed)
        return self.out_proj(attn)

    def _shape(self, x, batch: int, time: int):
        return x.reshape(batch, time, self.num_heads, self.head_dim).transpose(0, 2, 1, 3).reshape(batch * self.num_heads, time, self.head_dim)


class HubertEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = HubertAttention()
        self.self_attn_layer_norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.final_layer_norm = nn.LayerNorm(768)

    def __call__(self, x, padding_mask=None):
        residual = x
        x = self.self_attn(x, padding_mask)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.final_layer_norm(residual + x)


class HubertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_conv = nn.Sequential(WeightNormConv1d(), SamePad(), nn.GELU())
        self.layer_norm = nn.LayerNorm(768)
        self.layers = [HubertEncoderLayer() for _ in range(12)]

    def __call__(self, x, padding_mask=None, target_layer: int | None = None):
        if padding_mask is not None and bool(np.array(mx.any(padding_mask))):
            x = mx.where(padding_mask[..., None], 0, x)
        x = x + self.pos_conv(x)
        x = self.layer_norm(x)
        for index, layer in enumerate(self.layers):
            x = layer(x, padding_mask)
            if index == target_layer:
                break
        return x


class HubertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractionModel()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)
        self.encoder = HubertEncoder()
        self.final_proj = nn.Linear(768, 256)

    def extract_features(self, source, output_layer: int):
        features = self.feature_extractor(source)
        features = self.layer_norm(features)
        x = self.post_extract_proj(features)
        return self.encoder(x, padding_mask=None, target_layer=output_layer - 1)

    def infer(self, source, version: str):
        output_layer = 9 if version == "v1" else 12
        features = self.extract_features(source, output_layer=output_layer)
        if version == "v1":
            return self.final_proj(features)
        return features


def load_hubert(model_path: str | Path, device: str | None = None) -> HubertModel:
    resolve_device(device)
    model = HubertModel()
    weights = load_npz_weights(model_path) if str(model_path).endswith(".npz") else _load_torch_hubert_weights(model_path)
    validate_weight_keys(model, weights, "HuBERT")
    model.load_weights([(key, mx.array(value)) for key, value in weights], strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model


def extract_hubert_directory_mlx(
    exp_dir: str | Path,
    model_path: str | Path,
    version: str,
    device: str | None,
    batch_size: int = 16,
) -> None:
    from minirvc.preprocess.audio_io import load_audio_unfiltered

    exp_dir = Path(exp_dir)
    wav_dir = exp_dir / "1_16k_wavs"
    out_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_hubert(model_path, device=device)
    files = [path for path in sorted(wav_dir.iterdir()) if path.suffix == ".wav"]
    if batch_size > 1:
        items = [(path, load_audio_unfiltered(path, 16000)) for path in files]
        for group in _keyed_batches(items, batch_size, lambda item: item[1].shape[0]):
            _extract_hubert_batch(group, model, version, out_dir)
        return
    for wav_path in files:
        audio = mx.array(load_audio_unfiltered(wav_path, 16000), dtype=mx.float32).reshape(1, -1)
        features = model.infer(audio, version=version)
        mx.eval(features)
        np.save(out_dir / wav_path.name.replace(".wav", ".npy"), np.array(features[0], dtype=np.float32), allow_pickle=False)


def _extract_hubert_batch(group: list[tuple[Path, np.ndarray]], model: HubertModel, version: str, out_dir: Path) -> None:
    sample_lengths = [audio.shape[0] for _, audio in group]
    max_samples = max(sample_lengths)
    audio_batch = np.zeros((len(group), max_samples), dtype=np.float32)
    for index, (_, audio) in enumerate(group):
        audio_batch[index, : audio.shape[0]] = audio
    features = model.infer(mx.array(audio_batch, dtype=mx.float32), version=version)
    mx.eval(features)
    arrays = np.array(features, dtype=np.float32)
    for index, (path, _) in enumerate(group):
        np.save(out_dir / path.name.replace(".wav", ".npy"), arrays[index], allow_pickle=False)


def _load_torch_hubert_weights(path: str | Path) -> list[tuple[str, np.ndarray]]:
    checkpoint = load_torch_checkpoint(path, fairseq=True)
    state = checkpoint["model"]
    weights = []
    allowed_unexpected = {"mask_emb", "label_embs_concat"}
    for key, raw_value in state.items():
        if key in allowed_unexpected:
            continue
        value = as_numpy(raw_value)
        target = _map_hubert_key(key)
        if _is_hubert_conv1d_weight(key):
            value = conv1d_weight(value)
        elif value.dtype.kind == "f":
            value = float_array(value)
        weights.append((target, value))
    return weights


def _map_hubert_key(key: str) -> str:
    key = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.(\d+)\.", r"feature_extractor.conv_layers.\1.layers.\2.", key)
    key = re.sub(r"^encoder\.pos_conv\.(\d+)\.", r"encoder.pos_conv.layers.\1.", key)
    return key


def _is_hubert_conv1d_weight(key: str) -> bool:
    return key.endswith(".weight") and (
        re.match(r"^feature_extractor\.conv_layers\.\d+\.0\.weight$", key) is not None
    )


def _keyed_batches(items: list[tuple[Path, np.ndarray]], batch_size: int, key_fn) -> list[list[tuple[Path, np.ndarray]]]:
    buckets: dict[int, list[tuple[Path, np.ndarray]]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    groups: list[list[tuple[Path, np.ndarray]]] = []
    for _, bucket in sorted(buckets.items(), key=lambda pair: pair[0], reverse=True):
        groups.extend(bucket[index : index + batch_size] for index in range(0, len(bucket), batch_size))
    return groups
