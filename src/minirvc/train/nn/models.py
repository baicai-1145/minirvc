from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d, functional as F
from torch.nn.utils import spectral_norm, weight_norm

from minirvc.train.nn import attentions, commons, modules
from minirvc.train.nn.commons import get_padding, init_weights

_SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}
_HAS_XPU = bool(hasattr(torch, "xpu") and torch.xpu.is_available())


class TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = self.lrelu(x * math.sqrt(self.hidden_channels)).transpose(1, -1)
        x_mask = commons.sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        stats = self.proj(self.encoder(x * x_mask, x_mask)) * x_mask
        return torch.split(stats, self.out_channels, dim=1) + (x_mask,)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = commons.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        stats = self.proj(self.enc(self.pre(x) * x_mask, x_mask, g=g)) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel: int,
        upsample_kernel_sizes,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = _upsample_layers(upsample_rates, upsample_kernel_sizes, upsample_initial_channel)
        self.resblocks = _resblocks(
            resblock,
            self.num_upsamples,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )
        channels = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = self._upsample(x)
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def _upsample(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        for index, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, modules.LRELU_SLOPE))
            if noise is not None:
                x = x + self.noise_convs[index](noise)
            x = _apply_resblocks(self.resblocks, index, self.num_kernels, x)
        return x


class SineGen(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.dim = harmonic_num + 1
        self.sampling_rate = sample_rate
        self.voiced_threshold = voiced_threshold

    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = (f0 > self.voiced_threshold).to(dtype=f0.dtype)
            if uv.device.type == "privateuseone":
                uv = uv.float()
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            return sine_waves * uv + noise

    def _f02sine(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        frame_phase = f0 / self.sampling_rate * torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        accum = torch.fmod(frame_phase[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        accum = accum.cumsum(dim=1).fmod(1.0).to(f0)
        frame_phase = frame_phase + F.pad(accum, (0, 0, 1, 0), mode="constant")
        phase = frame_phase.reshape(f0.shape[0], -1, 1)
        phase = phase * torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        init_phase = torch.rand(1, 1, self.dim, device=f0.device)
        init_phase[..., 0] = 0
        return torch.sin(2 * np.pi * (phase + init_phase))


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0.0,
        is_half: bool = True,
    ):
        super().__init__()
        del is_half
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        return torch.tanh(self.l_linear(sine_wavs))


class GeneratorNSF(Generator):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel: int,
        upsample_kernel_sizes,
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ):
        super().__init__(
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.noise_convs = _noise_convs(upsample_rates, upsample_initial_channel)
        self.upp = math.prod(upsample_rates)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        return torch.tanh(self.conv_post(F.leaky_relu(self._upsample(x, har_source))))


def _upsample_layers(upsample_rates, upsample_kernel_sizes, initial_channel: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            weight_norm(
                ConvTranspose1d(
                    initial_channel // (2**index),
                    initial_channel // (2 ** (index + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
            )
            for index, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ]
    )


def _noise_convs(upsample_rates, initial_channel: int) -> nn.ModuleList:
    layers = []
    for index in range(len(upsample_rates)):
        channels = initial_channel // (2 ** (index + 1))
        if index + 1 < len(upsample_rates):
            stride = math.prod(upsample_rates[index + 1 :])
            layers.append(Conv1d(1, channels, kernel_size=stride * 2, stride=stride, padding=stride // 2))
        else:
            layers.append(Conv1d(1, channels, kernel_size=1))
    return nn.ModuleList(layers)


def _resblocks(
    resblock: str,
    num_upsamples: int,
    initial_channel: int,
    kernel_sizes,
    dilation_sizes,
) -> nn.ModuleList:
    block = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
    layers = nn.ModuleList()
    for index in range(num_upsamples):
        channels = initial_channel // (2 ** (index + 1))
        for kernel, dilation in zip(kernel_sizes, dilation_sizes):
            layers.append(block(channels, kernel, dilation))
    return layers


def _apply_resblocks(resblocks: nn.ModuleList, index: int, num_kernels: int, x: torch.Tensor) -> torch.Tensor:
    start = index * num_kernels
    out = resblocks[start](x)
    for block in resblocks[start + 1 : start + num_kernels]:
        out = out + block(x)
    return out / num_kernels


class _Synthesizer(nn.Module):
    def __init__(self, phone_channels: int, use_f0: bool, *args, **kwargs):
        super().__init__()
        self.use_f0 = use_f0
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
            self.dec = GeneratorNSF(
                *decoder_args,
                gin_channels=values["gin_channels"],
                sr=sample_rate,
                is_half=values.get("is_half", False),
            )
        else:
            self.dec = Generator(*decoder_args, gin_channels=values["gin_channels"])
        self.enc_q = PosteriorEncoder(
            values["spec_channels"],
            inter_channels,
            values["hidden_channels"],
            5,
            1,
            16,
            gin_channels=values["gin_channels"],
        )
        self.flow = ResidualCouplingBlock(inter_channels, values["hidden_channels"], 5, 1, 3, gin_channels=values["gin_channels"])
        self.emb_g = nn.Embedding(values["spk_embed_dim"], values["gin_channels"])

    def _posterior(self, phone, phone_lengths, y, y_lengths, ds, pitch=None):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        return g, z_slice, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def forward(self, phone, phone_lengths, *args):
        if self.use_f0:
            pitch, pitchf, y, y_lengths, ds = args
            g, z_slice, ids_slice, x_mask, y_mask, stats = self._posterior(phone, phone_lengths, y, y_lengths, ds, pitch)
            pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
            y_hat = self.dec(z_slice, pitchf, g=g)
        else:
            y, y_lengths, ds = args
            g, z_slice, ids_slice, x_mask, y_mask, stats = self._posterior(phone, phone_lengths, y, y_lengths, ds)
            y_hat = self.dec(z_slice, g=g)
        return y_hat, ids_slice, x_mask, y_mask, stats

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        pitch: torch.Tensor | None = None,
        pitchf: torch.Tensor | None = None,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        if self.use_f0:
            if pitchf is None:
                raise ValueError("pitchf is required for f0 inference")
            return self.dec(z * x_mask, pitchf, g=g)
        return self.dec(z * x_mask, g=g)


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
        phone_channels, values["inter_channels"], values["hidden_channels"], values["filter_channels"], values["n_heads"],
        values["n_layers"], values["kernel_size"], float(values["p_dropout"]), f0=use_f0,
    )


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = _period_discriminators([2, 3, 5, 7, 11, 17], use_spectral_norm)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


class MultiPeriodDiscriminatorV2(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        self.discriminators = _period_discriminators([2, 3, 5, 7, 11, 17, 23, 37], use_spectral_norm)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        return _run_discriminators(self.discriminators, y, y_hat)


def _period_discriminators(periods: list[int], use_spectral_norm: bool) -> nn.ModuleList:
    return nn.ModuleList([DiscriminatorS(use_spectral_norm)] + [DiscriminatorP(period, use_spectral_norm=use_spectral_norm) for period in periods])


def _run_discriminators(discriminators: nn.ModuleList, y: torch.Tensor, y_hat: torch.Tensor):
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
    for discriminator in discriminators:
        y_d_r, fmap_r = discriminator(y)
        y_d_g, fmap_g = discriminator(y_hat)
        y_d_rs.append(y_d_r)
        y_d_gs.append(y_d_g)
        fmap_rs.append(fmap_r)
        fmap_gs.append(fmap_g)
    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(Conv1d(in_ch, out_ch, kernel, stride, groups=groups, padding=padding))
                for in_ch, out_ch, kernel, stride, groups, padding in (
                    (1, 16, 15, 1, 1, 7),
                    (16, 64, 41, 4, 4, 20),
                    (64, 256, 41, 4, 16, 20),
                    (256, 1024, 41, 4, 64, 20),
                    (1024, 1024, 41, 4, 256, 20),
                    (1024, 1024, 5, 1, 1, 2),
                )
            ]
        )
        self.conv_post = norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(Conv2d(in_ch, out_ch, (kernel_size, 1), stride_value, padding=(get_padding(kernel_size, 1), 0)))
                for in_ch, out_ch, stride_value in (
                    (1, 32, (stride, 1)),
                    (32, 128, (stride, 1)),
                    (128, 512, (stride, 1)),
                    (512, 1024, (stride, 1)),
                    (1024, 1024, 1),
                )
            ]
        )
        self.conv_post = norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        batch, channels, time = x.shape
        if time % self.period:
            pad = self.period - (time % self.period)
            if _HAS_XPU and x.dtype == torch.bfloat16:
                x = F.pad(x.to(dtype=torch.float16), (0, pad), "reflect").to(dtype=torch.bfloat16)
            else:
                x = F.pad(x, (0, pad), "reflect")
            time += pad
        x = x.view(batch, channels, time // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap
