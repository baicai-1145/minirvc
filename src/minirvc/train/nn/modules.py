from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import Conv1d, functional as F
from torch.nn.utils import weight_norm

from minirvc.train.nn import commons
from minirvc.train.nn.commons import get_padding, init_weights

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.drop = nn.Dropout(float(p_dropout))
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels:
            self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight")

        for layer in range(n_layers):
            dilation = dilation_rate**layer
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding),
                    name="weight",
                )
            )
            res_skip_channels = 2 * hidden_channels if layer < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1), name="weight"))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)

        for layer, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            if g is None:
                g_l = torch.zeros_like(x_in)
            else:
                offset = layer * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels, :]
            acts = self.drop(commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels))
            res_skip = res_skip_layer(acts)
            if layer < self.n_layers - 1:
                x = (x + res_skip[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip[:, self.hidden_channels :, :]
            else:
                output = output + res_skip
        return output * x_mask


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))) for _ in dilation]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                residual = residual * x_mask
            residual = conv1(residual)
            residual = F.leaky_relu(residual, LRELU_SLOPE)
            if x_mask is not None:
                residual = residual * x_mask
            x = conv2(residual) + x
        return x if x_mask is None else x * x_mask


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv in self.convs:
            residual = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                residual = residual * x_mask
            x = conv(residual) + x
        return x if x_mask is None else x * x_mask


class Flip(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x_mask, g
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x, logdet


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=float(p_dropout),
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        m = self.post(h) * x_mask
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        if reverse:
            return torch.cat([x0, (x1 - m) * x_mask], 1), logdet

        return torch.cat([x0, m + x1 * x_mask], 1), logdet
