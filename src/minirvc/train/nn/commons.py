from __future__ import annotations

from typing import Optional

import torch


def init_weights(module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    out = torch.zeros_like(x[:, :, :segment_size])
    for index, start in enumerate(ids_str):
        out[index] = x[index, :, start : start + segment_size]
    return out


def slice_segments2(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    out = torch.zeros_like(x[:, :segment_size])
    for index, start in enumerate(ids_str):
        out[index] = x[index, start : start + segment_size]
    return out


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor | None = None,
    segment_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, time = x.size()
    if x_lengths is None:
        x_lengths = torch.full((batch,), time, dtype=torch.long, device=x.device)
    max_start = x_lengths - segment_size + 1
    ids = (torch.rand([batch]).to(device=x.device) * max_start).long()
    return slice_segments(x, ids, segment_size), ids


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    channels = n_channels[0]
    acts = input_a + input_b
    return torch.tanh(acts[:, :channels, :]) * torch.sigmoid(acts[:, channels:, :])


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    ids = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return ids.unsqueeze(0) < length.unsqueeze(1)


def clip_grad_value_(parameters, clip_value, norm_type: float = 2.0) -> float:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [param for param in parameters if param.grad is not None]
    total_norm = 0.0
    for param in grads:
        param_norm = param.grad.data.norm(float(norm_type))
        total_norm += param_norm.item() ** float(norm_type)
        if clip_value is not None:
            param.grad.data.clamp_(min=-float(clip_value), max=float(clip_value))
    return total_norm ** (1.0 / float(norm_type))
