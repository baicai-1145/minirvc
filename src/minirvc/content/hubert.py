from __future__ import annotations

from pathlib import Path
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self):
        super().__init__()
        specs = [(1, 512, 10, 5), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 3, 2), (512, 512, 2, 2), (512, 512, 2, 2)]
        layers = []
        for index, (in_channels, out_channels, kernel, stride) in enumerate(specs):
            if index == 0:
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False),
                        nn.Dropout(0.0),
                        nn.GroupNorm(out_channels, out_channels),
                        nn.GELU(),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False),
                        nn.Dropout(0.0),
                        nn.GELU(),
                    )
                )
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = source.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x


class WeightNormConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_g = nn.Parameter(torch.empty(1, 1, 128))
        self.weight_v = nn.Parameter(torch.empty(768, 48, 128))
        self.bias = nn.Parameter(torch.empty(768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(self.weight_v, dim=(0, 1), keepdim=True)
        weight = self.weight_v * (self.weight_g / norm)
        return F.conv1d(x, weight, self.bias, padding=64, groups=16)


class SamePad(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-1]


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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        time, batch, embed = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._shape(q, time, batch)
        k = self._shape(k, time, batch)
        v = self._shape(v, time, batch)

        weights = torch.bmm(q, k.transpose(1, 2))
        if padding_mask is not None and padding_mask.any():
            weights = weights.view(batch, self.num_heads, time, time)
            weights = weights.masked_fill(padding_mask[:, None, None, :], float("-inf"))
            weights = weights.view(batch * self.num_heads, time, time)
        probs = F.softmax(weights, dim=-1, dtype=torch.float32)
        attn = torch.bmm(probs, v)
        attn = attn.view(batch, self.num_heads, time, self.head_dim)
        attn = attn.permute(2, 0, 1, 3).reshape(time, batch, embed)
        return self.out_proj(attn)

    def _shape(self, x: torch.Tensor, time: int, batch: int) -> torch.Tensor:
        return x.contiguous().view(time, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(batch * self.num_heads, time, self.head_dim)


class HubertEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = HubertAttention()
        self.self_attn_layer_norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.final_layer_norm = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, padding_mask)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return self.final_layer_norm(x)


class HubertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_conv = nn.Sequential(WeightNormConv1d(), SamePad(), nn.GELU())
        self.layer_norm = nn.LayerNorm(768)
        self.layers = nn.ModuleList([HubertEncoderLayer() for _ in range(12)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None, target_layer: int | None) -> torch.Tensor:
        if padding_mask is not None and padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        for index, layer in enumerate(self.layers):
            x = layer(x, padding_mask)
            if index == target_layer:
                break
        return x.transpose(0, 1)


class HubertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractionModel()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)
        self.encoder = HubertEncoder()
        self.final_proj = nn.Linear(768, 256)

    def extract_features(self, source: torch.Tensor, output_layer: int) -> torch.Tensor:
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        x = self.post_extract_proj(features)
        target_layer = output_layer - 1
        return self.encoder(x, padding_mask=None, target_layer=target_layer)

    def infer(self, source: torch.Tensor, version: str) -> torch.Tensor:
        output_layer = 9 if version == "v1" else 12
        features = self.extract_features(source, output_layer=output_layer)
        if version == "v1":
            return self.final_proj(features)
        return features


def load_hubert(model_path: str | Path, device: str | torch.device) -> HubertModel:
    checkpoint = load_fairseq_checkpoint(model_path)
    model = HubertModel()
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_unexpected = {"mask_emb", "label_embs_concat"}
    if missing or set(unexpected) != allowed_unexpected:
        raise RuntimeError(
            f"HuBERT checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.float().eval().to(device)
    return model


def load_fairseq_checkpoint(model_path: str | Path) -> dict:
    class Dictionary:
        pass

    fairseq = types.ModuleType("fairseq")
    data = types.ModuleType("fairseq.data")
    dictionary = types.ModuleType("fairseq.data.dictionary")
    dictionary.Dictionary = Dictionary
    original = {name: sys.modules.get(name) for name in ("fairseq", "fairseq.data", "fairseq.data.dictionary")}
    sys.modules["fairseq"] = fairseq
    sys.modules["fairseq.data"] = data
    sys.modules["fairseq.data.dictionary"] = dictionary
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
