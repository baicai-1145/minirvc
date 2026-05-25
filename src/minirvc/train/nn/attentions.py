from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from minirvc.train.nn.modules import LayerNorm


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.n_layers = int(n_layers)
        self.drop = nn.Dropout(float(p_dropout))
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=float(p_dropout),
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=float(p_dropout),
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers,
            self.norm_layers_1,
            self.ffn_layers,
            self.norm_layers_2,
        ):
            x = norm1(x + self.drop(attn(x, x, attn_mask)))
            x = norm2(x + self.drop(ffn(x, x_mask)))
        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(float(p_dropout))

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(context)
        v = self.conv_v(context)
        return self.conv_o(self.attention(q, k, v, attn_mask))

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, channels, source_len = key.size()
        target_len = query.size(2)
        query = query.view(batch, self.n_heads, self.k_channels, target_len).transpose(2, 3)
        key = key.view(batch, self.n_heads, self.k_channels, source_len).transpose(2, 3)
        value = value.view(batch, self.n_heads, self.k_channels, source_len).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        if self.window_size is not None:
            rel_k = self._get_relative_embeddings(self.emb_rel_k, source_len)
            scores = scores + self._relative_position_to_absolute_position(
                torch.matmul(query, rel_k.unsqueeze(0).transpose(-2, -1))
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, value)
        if self.window_size is not None:
            rel_v = self._get_relative_embeddings(self.emb_rel_v, source_len)
            rel_attn = self._absolute_position_to_relative_position(attn)
            output = output + torch.matmul(rel_attn, rel_v.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(batch, channels, target_len)

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        return relative_embeddings[:, slice_start:slice_end]

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x = x.view(batch, heads, length * 2 * length)
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0])
        return x.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x = x.view(batch, heads, length**2 + length * (length - 1))
        x = F.pad(x, [length, 0, 0, 0, 0, 0])
        return x.view(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(float(p_dropout))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self._same_padding(x * x_mask))
        x = self.drop(torch.relu(x))
        x = self.conv_2(self._same_padding(x * x_mask))
        return x * x_mask

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        return F.pad(x, [pad_left, pad_right, 0, 0, 0, 0])
