from __future__ import annotations

import math
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn


LRELU_SLOPE = 0.1
_SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}


def leaky_relu(x, slope: float = LRELU_SLOPE):
    return mx.maximum(x, x * slope)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _weight_norm(weight_v, weight_g, axes):
    norm = mx.sqrt(mx.sum(weight_v * weight_v, axis=axes, keepdims=True)) + 1e-12
    return weight_v * (weight_g / norm)


class Conv1dNCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1 / max(1, in_channels * kernel_size))
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(out_channels, kernel_size, in_channels // groups))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)

    def __call__(self, x):
        y = mx.conv1d(x.swapaxes(1, 2), self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class WeightNormConv1dNCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        raw_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight_v = mx.random.normal(raw_shape) * 0.01
        self.weight_g = mx.sqrt(mx.sum(self.weight_v * self.weight_v, axis=(1, 2), keepdims=True))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)

    def __call__(self, x):
        weight = _weight_norm(self.weight_v, self.weight_g, (1, 2)).transpose(0, 2, 1)
        y = mx.conv1d(x.swapaxes(1, 2), weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class ConvTranspose1dNCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1 / max(1, in_channels * kernel_size))
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.output_padding = int(output_padding)

    def __call__(self, x):
        y = mx.conv_transpose1d(
            x.swapaxes(1, 2),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
        )
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class WeightNormConvTranspose1dNCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, bias: bool = True):
        super().__init__()
        raw_shape = (in_channels, out_channels, kernel_size)
        self.weight_v = mx.random.normal(raw_shape) * 0.01
        self.weight_g = mx.sqrt(mx.sum(self.weight_v * self.weight_v, axis=(1, 2), keepdims=True))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)

    def __call__(self, x):
        weight = _weight_norm(self.weight_v, self.weight_g, (1, 2)).transpose(1, 2, 0)
        y = mx.conv_transpose1d(x.swapaxes(1, 2), weight, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(1, 2)


class WeightNormConv2dNCHW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, groups: int = 1, bias: bool = True):
        super().__init__()
        kh, kw = _pair(kernel_size)
        raw_shape = (out_channels, in_channels // groups, kh, kw)
        self.weight_v = mx.random.normal(raw_shape) * 0.01
        self.weight_g = mx.sqrt(mx.sum(self.weight_v * self.weight_v, axis=(1, 2, 3), keepdims=True))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = int(groups)

    def __call__(self, x):
        weight = _weight_norm(self.weight_v, self.weight_g, (1, 2, 3)).transpose(0, 2, 3, 1)
        y = mx.conv2d(x.transpose(0, 2, 3, 1), weight, stride=self.stride, padding=self.padding, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(0, 3, 1, 2)


class LayerNormNCL(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        y = x.swapaxes(1, -1)
        mean = mx.mean(y, axis=-1, keepdims=True)
        var = mx.var(y, axis=-1, keepdims=True)
        y = (y - mean) * mx.rsqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y.swapaxes(1, -1)


class WN(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0, p_dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.n_layers = int(n_layers)
        self.gin_channels = int(gin_channels)
        self.drop = nn.Dropout(float(p_dropout))
        self.in_layers = []
        self.res_skip_layers = []
        if gin_channels:
            self.cond_layer = WeightNormConv1dNCL(gin_channels, 2 * hidden_channels * n_layers, 1)
        for layer in range(n_layers):
            dilation = dilation_rate**layer
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(WeightNormConv1dNCL(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding))
            res_skip_channels = 2 * hidden_channels if layer < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(WeightNormConv1dNCL(hidden_channels, res_skip_channels, 1))

    def __call__(self, x, x_mask, g=None):
        output = mx.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        for layer, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            if g is None:
                g_l = mx.zeros_like(x_in)
            else:
                offset = layer * 2 * self.hidden_channels
                g_l = g[:, offset : offset + 2 * self.hidden_channels, :]
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels))
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
        self.convs1 = [WeightNormConv1dNCL(channels, channels, kernel_size, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation]
        self.convs2 = [WeightNormConv1dNCL(channels, channels, kernel_size, dilation=1, padding=get_padding(kernel_size, 1)) for _ in dilation]

    def __call__(self, x, x_mask=None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = leaky_relu(x)
            if x_mask is not None:
                residual = residual * x_mask
            residual = conv1(residual)
            residual = leaky_relu(residual)
            if x_mask is not None:
                residual = residual * x_mask
            x = conv2(residual) + x
        return x if x_mask is None else x * x_mask


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3)):
        super().__init__()
        self.convs = [WeightNormConv1dNCL(channels, channels, kernel_size, dilation=d, padding=get_padding(kernel_size, d)) for d in dilation]

    def __call__(self, x, x_mask=None):
        for conv in self.convs:
            residual = leaky_relu(x)
            if x_mask is not None:
                residual = residual * x_mask
            x = conv(residual) + x
        return x if x_mask is None else x * x_mask


class Flip(nn.Module):
    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        del x_mask, g, reverse
        return x[:, ::-1, :], mx.zeros((x.shape[0],), dtype=x.dtype)


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, p_dropout: float = 0.0, gin_channels: int = 0):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.pre = Conv1dNCL(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=float(p_dropout), gin_channels=gin_channels)
        self.post = Conv1dNCL(hidden_channels, self.half_channels, 1)
        self.post.weight = mx.zeros_like(self.post.weight)
        self.post.bias = mx.zeros_like(self.post.bias)

    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        x0, x1 = mx.split(x, 2, axis=1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        m = self.post(h) * x_mask
        logdet = mx.zeros((x.shape[0],), dtype=x.dtype)
        if reverse:
            return mx.concatenate([x0, (x1 - m) * x_mask], axis=1), logdet
        return mx.concatenate([x0, m + x1 * x_mask], axis=1), logdet


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int, n_heads: int, p_dropout: float = 0.0, window_size: int | None = None, heads_share: bool = True):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = int(n_heads)
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = Conv1dNCL(channels, channels, 1)
        self.conv_k = Conv1dNCL(channels, channels, 1)
        self.conv_v = Conv1dNCL(channels, channels, 1)
        self.conv_o = Conv1dNCL(channels, out_channels, 1)
        self.drop = nn.Dropout(float(p_dropout))
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = mx.random.normal((n_heads_rel, window_size * 2 + 1, self.k_channels)) * rel_stddev
            self.emb_rel_v = mx.random.normal((n_heads_rel, window_size * 2 + 1, self.k_channels)) * rel_stddev

    def __call__(self, x, context, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(context)
        v = self.conv_v(context)
        return self.conv_o(self.attention(q, k, v, attn_mask))

    def attention(self, query, key, value, mask):
        batch, channels, source_len = key.shape
        target_len = query.shape[2]
        query = query.reshape(batch, self.n_heads, self.k_channels, target_len).swapaxes(2, 3)
        key = key.reshape(batch, self.n_heads, self.k_channels, source_len).swapaxes(2, 3)
        value = value.reshape(batch, self.n_heads, self.k_channels, source_len).swapaxes(2, 3)
        query = query / math.sqrt(self.k_channels)
        scores = query @ key.swapaxes(-2, -1)
        if self.window_size is not None:
            rel_k = self._get_relative_embeddings(self.emb_rel_k, source_len)
            scores = scores + self._relative_position_to_absolute_position(query @ rel_k[None].swapaxes(-2, -1))
        if mask is not None:
            scores = mx.where(mask == 0, -1e4, scores)
        attn = self.drop(mx.softmax(scores, axis=-1))
        output = attn @ value
        if self.window_size is not None:
            rel_v = self._get_relative_embeddings(self.emb_rel_v, source_len)
            rel_attn = self._absolute_position_to_relative_position(attn)
            output = output + rel_attn @ rel_v[None]
        return output.swapaxes(2, 3).reshape(batch, channels, target_len)

    def _get_relative_embeddings(self, relative_embeddings, length: int):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            relative_embeddings = mx.pad(relative_embeddings, [(0, 0), (pad_length, pad_length), (0, 0)])
        return relative_embeddings[:, slice_start:slice_end]

    @staticmethod
    def _relative_position_to_absolute_position(x):
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1)])
        x = x.reshape(batch, heads, length * 2 * length)
        x = mx.pad(x, [(0, 0), (0, 0), (0, length - 1)])
        return x.reshape(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(x):
        batch, heads, length, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, length - 1)])
        x = x.reshape(batch, heads, length**2 + length * (length - 1))
        x = mx.pad(x, [(0, 0), (0, 0), (length, 0)])
        return x.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int, kernel_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv_1 = Conv1dNCL(in_channels, filter_channels, kernel_size)
        self.conv_2 = Conv1dNCL(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(float(p_dropout))

    def __call__(self, x, x_mask):
        x = self.conv_1(self._same_padding(x * x_mask))
        x = self.drop(mx.maximum(x, 0))
        x = self.conv_2(self._same_padding(x * x_mask))
        return x * x_mask

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        return mx.pad(x, [(0, 0), (0, 0), (pad_left, pad_right)])


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int, kernel_size: int = 1, p_dropout: float = 0.0, window_size: int = 10, **kwargs):
        super().__init__()
        del kwargs
        self.n_layers = int(n_layers)
        self.drop = nn.Dropout(float(p_dropout))
        self.attn_layers = []
        self.norm_layers_1 = []
        self.ffn_layers = []
        self.norm_layers_2 = []
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=float(p_dropout), window_size=window_size))
            self.norm_layers_1.append(LayerNormNCL(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=float(p_dropout)))
            self.norm_layers_2.append(LayerNormNCL(hidden_channels))

    def __call__(self, x, x_mask):
        attn_mask = x_mask[:, :, None, :] * x_mask[:, :, :, None]
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = norm1(x + self.drop(attn(x, x, attn_mask)))
            x = norm2(x + self.drop(ffn(x, x_mask)))
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int, kernel_size: int, p_dropout: float, f0: bool = True):
        super().__init__()
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout))
        self.proj = Conv1dNCL(hidden_channels, out_channels * 2, 1)

    def __call__(self, phone, pitch, lengths, skip_head=None):
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = leaky_relu(x * math.sqrt(self.hidden_channels)).swapaxes(1, 2)
        x_mask = sequence_mask(lengths, x.shape[2])[:, None, :].astype(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            head = _scalar_int(skip_head)
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats = self.proj(x) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        return m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, n_flows: int = 4, gin_channels: int = 0):
        super().__init__()
        self.flows = []
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels))
            self.flows.append(Flip())

    def __call__(self, x, x_mask, g=None, reverse: bool = False):
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0):
        super().__init__()
        self.out_channels = int(out_channels)
        self.pre = Conv1dNCL(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = Conv1dNCL(hidden_channels, out_channels * 2, 1)

    def __call__(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.shape[2])[:, None, :].astype(x.dtype)
        stats = self.proj(self.enc(self.pre(x) * x_mask, x_mask, g=g)) * x_mask
        m, logs = mx.split(stats, 2, axis=1)
        z = (m + mx.random.normal(m.shape) * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    def __init__(self, initial_channel: int, resblock: str, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel: int, upsample_kernel_sizes, gin_channels: int = 0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1dNCL(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = _upsample_layers(upsample_rates, upsample_kernel_sizes, upsample_initial_channel)
        self.resblocks = _resblocks(resblock, self.num_upsamples, upsample_initial_channel, resblock_kernel_sizes, resblock_dilation_sizes)
        channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = Conv1dNCL(channels, 1, 7, 1, padding=3, bias=False)
        if gin_channels:
            self.cond = Conv1dNCL(gin_channels, upsample_initial_channel, 1)

    def __call__(self, x, g=None, n_res=None):
        del n_res
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self._upsample(x)
        return mx.tanh(self.conv_post(leaky_relu(x, 0.01)))

    def _upsample(self, x, noise=None):
        for index, upsample in enumerate(self.ups):
            x = upsample(leaky_relu(x))
            if noise is not None:
                x = x + self.noise_convs[index](noise)
            x = _apply_resblocks(self.resblocks, index, self.num_kernels, x)
        return x


class SineGen(nn.Module):
    def __init__(self, sample_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.sine_amp = float(sine_amp)
        self.noise_std = float(noise_std)
        self.dim = int(harmonic_num) + 1
        self.sampling_rate = int(sample_rate)
        self.voiced_threshold = float(voiced_threshold)

    def __call__(self, f0, upp: int):
        f0 = f0[..., None]
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).astype(f0.dtype)
        uv = mx.repeat(uv, repeats=int(upp), axis=1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        return sine_waves * uv + noise

    def _f02sine(self, f0, upp: int):
        steps = mx.arange(1, upp + 1, dtype=f0.dtype)
        frame_phase = f0 / self.sampling_rate * steps.reshape(1, 1, -1)
        accum = mx.remainder(frame_phase[:, :-1, -1:].astype(mx.float32) + 0.5, 1.0) - 0.5
        accum = mx.remainder(mx.cumsum(accum, axis=1), 1.0).astype(f0.dtype)
        accum = mx.pad(accum, [(0, 0), (1, 0), (0, 0)])
        frame_phase = frame_phase + accum
        phase = frame_phase.reshape(f0.shape[0], -1, 1)
        phase = phase * mx.arange(1, self.dim + 1, dtype=f0.dtype).reshape(1, 1, -1)
        return mx.sin(2 * np.pi * phase)


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshod: float = 0.0, is_half: bool = True):
        super().__init__()
        del is_half
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x, upp: int = 1):
        sine_wavs = self.l_sin_gen(x, upp)
        return mx.tanh(self.l_linear(sine_wavs))


class GeneratorNSF(Generator):
    def __init__(self, initial_channel: int, resblock: str, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel: int, upsample_kernel_sizes, gin_channels: int, sr: int, is_half: bool = False):
        super().__init__(initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.noise_convs = _noise_convs(upsample_rates, upsample_initial_channel)
        self.upp = math.prod(upsample_rates)

    def __call__(self, x, f0, g=None, n_res=None):
        del n_res
        har_source = self.m_source(f0, self.upp).swapaxes(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        return mx.tanh(self.conv_post(leaky_relu(self._upsample(x, har_source), 0.01)))


class _Synthesizer(nn.Module):
    def __init__(self, phone_channels: int, use_f0: bool, *args, **kwargs):
        super().__init__()
        self.use_f0 = bool(use_f0)
        values = _synth_kwargs(args, kwargs)
        self.segment_size = values["segment_size"]
        self.enc_p = _text_encoder(phone_channels, values, use_f0)
        inter_channels = values["inter_channels"]
        decoder_args = (
            inter_channels,
            values["resblock"],
            values["resblock_kernel_sizes"],
            values["resblock_dilation_sizes"],
            values["upsample_rates"],
            values["upsample_initial_channel"],
            values["upsample_kernel_sizes"],
        )
        if use_f0:
            sr = values["sr"]
            sample_rate = _SAMPLE_RATES[sr] if isinstance(sr, str) else int(sr)
            self.dec = GeneratorNSF(*decoder_args, gin_channels=values["gin_channels"], sr=sample_rate, is_half=values.get("is_half", False))
        else:
            self.dec = Generator(*decoder_args, gin_channels=values["gin_channels"])
        self.enc_q = PosteriorEncoder(values["spec_channels"], inter_channels, values["hidden_channels"], 5, 1, 16, gin_channels=values["gin_channels"])
        self.flow = ResidualCouplingBlock(inter_channels, values["hidden_channels"], 5, 1, 3, gin_channels=values["gin_channels"])
        self.emb_g = nn.Embedding(values["spk_embed_dim"], values["gin_channels"])

    def _speaker_condition(self, ds):
        g = self.emb_g(ds)
        if g.ndim == 2:
            return g[:, :, None]
        return g.swapaxes(1, 2)

    def _posterior(self, phone, phone_lengths, y, y_lengths, ds, pitch=None):
        g = self._speaker_condition(ds)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        return g, z_slice, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def __call__(self, phone, phone_lengths, *args):
        if self.use_f0:
            pitch, pitchf, y, y_lengths, ds = args
            g, z_slice, ids_slice, x_mask, y_mask, stats = self._posterior(phone, phone_lengths, y, y_lengths, ds, pitch)
            pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
            y_hat = self.dec(z_slice, pitchf, g=g)
        else:
            y, y_lengths, ds = args
            g, z_slice, ids_slice, x_mask, y_mask, stats = self._posterior(phone, phone_lengths, y, y_lengths, ds)
            y_hat = self.dec(z_slice, g=g)
        return y_hat, ids_slice, x_mask, y_mask, stats

    def infer(self, phone, phone_lengths, sid, pitch=None, pitchf=None, noise_scale: float = 0.66666, skip_head=None, return_length=None, return_length2=None):
        g = self._speaker_condition(sid)
        if skip_head is not None and return_length is not None:
            head = _scalar_int(skip_head)
            length = _scalar_int(return_length)
            flow_head = max(head - 24, 0)
            dec_head = head - flow_head
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
        else:
            head = length = dec_head = None
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + mx.exp(logs_p) * mx.random.normal(m_p.shape) * float(noise_scale)) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        if length is not None:
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
        if self.use_f0:
            if pitchf is None:
                raise ValueError("pitchf is required for f0 inference")
            if length is not None:
                pitchf = pitchf[:, head : head + length]
            return self.dec(z * x_mask, pitchf, g=g, n_res=return_length2)
        return self.dec(z * x_mask, g=g, n_res=return_length2)


class SynthesizerTrnMs256NSFsid(_Synthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(256, True, *args, **kwargs)


class SynthesizerTrnMs768NSFsid(_Synthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(768, True, *args, **kwargs)


class SynthesizerTrnMs256NSFsid_nono(_Synthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(256, False, *args, **kwargs)


class SynthesizerTrnMs768NSFsid_nono(_Synthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(768, False, *args, **kwargs)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        if use_spectral_norm:
            raise ValueError("MLX discriminator does not support spectral_norm yet")
        self.discriminators = _period_discriminators([2, 3, 5, 7, 11, 17])

    def __call__(self, y, y_hat):
        return _run_discriminators(self.discriminators, y, y_hat)


class MultiPeriodDiscriminatorV2(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        if use_spectral_norm:
            raise ValueError("MLX discriminator does not support spectral_norm yet")
        self.discriminators = _period_discriminators([2, 3, 5, 7, 11, 17, 23, 37])

    def __call__(self, y, y_hat):
        return _run_discriminators(self.discriminators, y, y_hat)


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = [
            WeightNormConv1dNCL(in_ch, out_ch, kernel, stride, groups=groups, padding=padding)
            for in_ch, out_ch, kernel, stride, groups, padding in (
                (1, 16, 15, 1, 1, 7),
                (16, 64, 41, 4, 4, 20),
                (64, 256, 41, 4, 16, 20),
                (256, 1024, 41, 4, 64, 20),
                (1024, 1024, 41, 4, 256, 20),
                (1024, 1024, 5, 1, 1, 2),
            )
        ]
        self.conv_post = WeightNormConv1dNCL(1024, 1, 3, 1, padding=1)

    def __call__(self, x):
        fmap = []
        for conv in self.convs:
            x = leaky_relu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.reshape(x.shape[0], -1), fmap


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = int(period)
        self.convs = [
            WeightNormConv2dNCHW(in_ch, out_ch, (kernel_size, 1), stride_value, padding=(get_padding(kernel_size, 1), 0))
            for in_ch, out_ch, stride_value in (
                (1, 32, (stride, 1)),
                (32, 128, (stride, 1)),
                (128, 512, (stride, 1)),
                (512, 1024, (stride, 1)),
                (1024, 1024, 1),
            )
        ]
        self.conv_post = WeightNormConv2dNCHW(1024, 1, (3, 1), 1, padding=(1, 0))

    def __call__(self, x):
        fmap = []
        batch, channels, time = x.shape
        if time % self.period:
            pad = self.period - (time % self.period)
            x = reflect_pad1d_ncl(x, 0, pad)
            time += pad
        x = x.reshape(batch, channels, time // self.period, self.period)
        for conv in self.convs:
            x = leaky_relu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.reshape(x.shape[0], -1), fmap


def build_generator(hps, use_f0: bool, version: str, sample_rate: str) -> nn.Module:
    cls = {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }[(version, use_f0)]
    args = (hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length)
    if use_f0:
        return cls(*args, **vars(hps.model), is_half=False, sr=sample_rate)
    return cls(*args, **vars(hps.model), is_half=False)


def build_discriminator(hps, version: str) -> nn.Module:
    cls = MultiPeriodDiscriminator if version == "v1" else MultiPeriodDiscriminatorV2
    return cls(hps.model.use_spectral_norm)


def sequence_mask(length, max_length: Optional[int] = None):
    if max_length is None:
        max_length = int(np.array(mx.max(length)))
    ids = mx.arange(max_length, dtype=length.dtype)
    return ids[None, :] < length[:, None]


def slice_segments(x, ids_str, segment_size: int):
    starts = np.array(ids_str).astype(int).reshape(-1)
    return mx.stack([x[index, :, start : start + segment_size] for index, start in enumerate(starts)], axis=0)


def slice_segments2(x, ids_str, segment_size: int):
    starts = np.array(ids_str).astype(int).reshape(-1)
    return mx.stack([x[index, start : start + segment_size] for index, start in enumerate(starts)], axis=0)


def rand_slice_segments(x, x_lengths=None, segment_size: int = 4):
    batch, _, time = x.shape
    if x_lengths is None:
        x_lengths = mx.full((batch,), time, dtype=mx.int32)
    max_start = np.maximum(np.array(x_lengths).astype(int) - segment_size + 1, 1)
    ids_np = (np.random.random((batch,)) * max_start).astype(np.int32)
    ids = mx.array(ids_np, dtype=mx.int32)
    return slice_segments(x, ids, segment_size), ids


def fused_add_tanh_sigmoid_multiply(input_a, input_b, channels: int):
    acts = input_a + input_b
    return mx.tanh(acts[:, :channels, :]) * mx.sigmoid(acts[:, channels:, :])


def reflect_pad1d_ncl(x, left: int, right: int):
    pieces = []
    if left:
        pieces.append(x[:, :, 1 : left + 1][:, :, ::-1])
    pieces.append(x)
    if right:
        pieces.append(x[:, :, -right - 1 : -1][:, :, ::-1])
    return mx.concatenate(pieces, axis=2)


def _upsample_layers(upsample_rates, upsample_kernel_sizes, initial_channel: int):
    return [
        WeightNormConvTranspose1dNCL(
            initial_channel // (2**index),
            initial_channel // (2 ** (index + 1)),
            kernel,
            rate,
            padding=(kernel - rate) // 2,
        )
        for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
    ]


def _noise_convs(upsample_rates, initial_channel: int):
    layers = []
    for index in range(len(upsample_rates)):
        channels = initial_channel // (2 ** (index + 1))
        if index + 1 < len(upsample_rates):
            stride = math.prod(upsample_rates[index + 1 :])
            layers.append(Conv1dNCL(1, channels, kernel_size=stride * 2, stride=stride, padding=stride // 2))
        else:
            layers.append(Conv1dNCL(1, channels, kernel_size=1))
    return layers


def _resblocks(resblock: str, num_upsamples: int, initial_channel: int, kernel_sizes, dilation_sizes):
    block = ResBlock1 if resblock == "1" else ResBlock2
    layers = []
    for index in range(num_upsamples):
        channels = initial_channel // (2 ** (index + 1))
        for kernel, dilation in zip(kernel_sizes, dilation_sizes):
            layers.append(block(channels, kernel, dilation))
    return layers


def _apply_resblocks(resblocks, index: int, num_kernels: int, x):
    start = index * num_kernels
    out = resblocks[start](x)
    for block in resblocks[start + 1 : start + num_kernels]:
        out = out + block(x)
    return out / num_kernels


def _period_discriminators(periods: list[int]):
    return [DiscriminatorS()] + [DiscriminatorP(period) for period in periods]


def _run_discriminators(discriminators, y, y_hat):
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
    for discriminator in discriminators:
        y_d_r, fmap_r = discriminator(y)
        y_d_g, fmap_g = discriminator(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def _synth_kwargs(args, kwargs) -> dict:
    names = (
        "spec_channels", "segment_size", "inter_channels", "hidden_channels", "filter_channels", "n_heads",
        "n_layers", "kernel_size", "p_dropout", "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
        "upsample_rates", "upsample_initial_channel", "upsample_kernel_sizes", "spk_embed_dim", "gin_channels", "sr",
    )
    values = dict(zip(names, args))
    values.update(kwargs)
    return values


def _text_encoder(phone_channels: int, values: dict, use_f0: bool) -> TextEncoder:
    return TextEncoder(
        phone_channels,
        values["inter_channels"],
        values["hidden_channels"],
        values["filter_channels"],
        values["n_heads"],
        values["n_layers"],
        values["kernel_size"],
        float(values["p_dropout"]),
        f0=use_f0,
    )


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _scalar_int(value) -> int:
    return int(np.array(value).reshape(-1)[0])
