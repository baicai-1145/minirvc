from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from minirvc.mlx.device import resolve_device
from minirvc.mlx.train_checkpoint import (
    export_small_model,
    latest_checkpoint_path,
    load_pretrained,
    load_training_checkpoint,
    save_training_checkpoint,
)
from minirvc.mlx.train_data import RvcMlxDataset
from minirvc.mlx.train_losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from minirvc.mlx.train_mel import mel_spectrogram, spec_to_mel
from minirvc.mlx.train_models import build_discriminator, build_generator, slice_segments
from minirvc.train.config import load_config, to_namespace


def train(
    *,
    exp_dir: str | Path,
    filelist: str | Path,
    version: str,
    sample_rate: str,
    use_f0: bool,
    batch_size: int,
    epochs: int,
    save_every_epoch: int,
    pretrain_g: str | Path,
    pretrain_d: str | Path,
    device: str | None = None,
    config_json: str | Path | None = None,
    num_workers: int = 0,
    resume: bool = True,
    export_final: bool = True,
    precision: str = "fp32",
) -> None:
    resolve_device(device)
    compute_dtype = _resolve_precision(precision)
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(exp_dir)
    if num_workers:
        logger.info("MLX training loads batches in-process; ignoring workers=%s", num_workers)
    config = load_config(version, sample_rate, config_json)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["data"]["training_files"] = str(filelist)
    hps = to_namespace(config)
    hps.model_dir = str(exp_dir)
    hps.name = exp_dir.name
    hps.sample_rate = sample_rate
    hps.if_f0 = int(use_f0)
    hps.version = version

    np.random.seed(int(hps.train.seed))
    dataset = RvcMlxDataset(filelist, hps.data, use_f0)
    net_g = build_generator(hps, use_f0, version, sample_rate)
    net_d = build_discriminator(hps, version)

    learning_rate = float(hps.train.learning_rate)
    start_epoch = 1
    global_step = 0
    latest_g = latest_checkpoint_path(exp_dir, "G_*.mlx.npz") if resume else None
    latest_d = latest_checkpoint_path(exp_dir, "D_*.mlx.npz") if resume else None
    if latest_g is not None and latest_d is not None:
        g_step, _, g_epoch = load_training_checkpoint(latest_g, net_g, "generator")
        d_step, _, d_epoch = load_training_checkpoint(latest_d, net_d, "discriminator")
        start_epoch = min(g_epoch, d_epoch) + 1
        global_step = min(g_step, d_step)
        logger.info("resumed MLX model weights from epoch %s; optimizer state was reset", start_epoch - 1)
    else:
        load_pretrained(pretrain_g, net_g, "generator")
        load_pretrained(pretrain_d, net_d, "discriminator")
        logger.info("loaded pretrained %s", pretrain_g)
        logger.info("loaded pretrained %s", pretrain_d)

    if compute_dtype != mx.float32:
        net_g.set_dtype(compute_dtype)
        net_d.set_dtype(compute_dtype)
        logger.info("using MLX mixed precision: %s", precision)

    optim_g = optim.AdamW(learning_rate=learning_rate, betas=list(hps.train.betas), eps=float(hps.train.eps), weight_decay=0.0)
    optim_d = optim.AdamW(learning_rate=learning_rate, betas=list(hps.train.betas), eps=float(hps.train.eps), weight_decay=0.0)
    mx.eval(net_g.parameters(), net_d.parameters(), optim_g.state, optim_d.state)

    for epoch in range(start_epoch, epochs + 1):
        epoch_lr = learning_rate * (float(hps.train.lr_decay) ** max(epoch - 1, 0))
        optim_g.learning_rate = mx.array(epoch_lr, dtype=mx.float32)
        optim_d.learning_rate = mx.array(epoch_lr, dtype=mx.float32)
        epoch_start = time.perf_counter()
        for batch_idx, batch in enumerate(dataset.batches(batch_size, epoch)):
            metrics = _train_step(batch, hps, use_f0, net_g, net_d, optim_g, optim_d, compute_dtype)
            if global_step % hps.train.log_interval == 0:
                logger.info("Train Epoch: %s [batch=%s]", epoch, batch_idx)
                logger.info([global_step, epoch_lr])
                logger.info(
                    "loss_disc=%.3f, loss_gen=%.3f, loss_fm=%.3f, loss_mel=%.3f, loss_kl=%.3f",
                    metrics["loss_disc"],
                    metrics["loss_gen"],
                    metrics["loss_fm"],
                    metrics["loss_mel"],
                    metrics["loss_kl"],
                )
            global_step += 1
        if epoch % save_every_epoch == 0:
            save_training_checkpoint(net_g, exp_dir / f"G_{global_step}.mlx.npz", iteration=global_step, learning_rate=epoch_lr, epoch=epoch)
            save_training_checkpoint(net_d, exp_dir / f"D_{global_step}.mlx.npz", iteration=global_step, learning_rate=epoch_lr, epoch=epoch)
        logger.info("====> Epoch: %s [%s] epoch_time=%.3fs", epoch, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.perf_counter() - epoch_start)

    if export_final:
        export_small_model(net_g, exp_dir / f"{exp_dir.name}.mlx.npz", hps, sample_rate, use_f0, version, epochs)
        logger.info("saving final mlx ckpt:Success.")


def _train_step(batch, hps, use_f0: bool, net_g, net_d, optim_g, optim_d, compute_dtype) -> dict[str, float]:
    data = _to_mx_batch(batch, use_f0, compute_dtype)
    if use_f0:
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, sid = data
    else:
        phone, phone_lengths, spec, spec_lengths, wave, sid = data
        pitch = pitchf = None

    mel = spec_to_mel(spec.astype(mx.float32), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)

    def forward_generator(gen):
        if use_f0:
            return gen(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
        return gen(phone, phone_lengths, spec, spec_lengths, sid)

    y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = forward_generator(net_g)
    y_hat_detached = mx.stop_gradient(y_hat)
    wave_seg = slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)

    def d_loss_fn(disc):
        y_d_hat_r, y_d_hat_g, _, _ = disc(wave_seg, y_hat_detached)
        return discriminator_loss(y_d_hat_r, y_d_hat_g)

    loss_disc, grads_d = nn.value_and_grad(net_d, d_loss_fn)(net_d)
    optim_d.update(net_d, grads_d)
    mx.eval(net_d.parameters(), optim_d.state)

    def g_loss_fn(gen):
        y_hat_g, ids_slice_g, _, z_mask_g, (_, z_p_g, m_p_g, logs_p_g, _, logs_q_g) = forward_generator(gen)
        y_mel = slice_segments(mel, ids_slice_g, hps.train.segment_size // hps.data.hop_length)
        wave_seg_g = slice_segments(wave, ids_slice_g * hps.data.hop_length, hps.train.segment_size)
        y_hat_mel = mel_spectrogram(
            y_hat_g[:, 0, :].astype(mx.float32),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_seg_g, y_hat_g)
        loss_mel = mx.mean(mx.abs(y_mel.astype(mx.float32) - y_hat_mel)) * hps.train.c_mel
        loss_kl = kl_loss(z_p_g, logs_q_g, m_p_g, logs_p_g, z_mask_g) * hps.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(y_d_hat_g)
        return loss_gen + loss_fm + loss_mel + loss_kl

    loss_gen_all, grads_g = nn.value_and_grad(net_g, g_loss_fn)(net_g)
    optim_g.update(net_g, grads_g)
    mx.eval(net_g.parameters(), optim_g.state)

    # Recompute scalar components for logging without keeping grads.
    y_hat_log, ids_log, _, z_mask_log, (_, z_p_log, m_p_log, logs_p_log, _, logs_q_log) = forward_generator(net_g)
    y_mel_log = slice_segments(mel, ids_log, hps.train.segment_size // hps.data.hop_length)
    wave_seg_log = slice_segments(wave, ids_log * hps.data.hop_length, hps.train.segment_size)
    y_hat_mel_log = mel_spectrogram(
        y_hat_log[:, 0, :].astype(mx.float32),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_seg_log, y_hat_log)
    loss_mel_log = mx.mean(mx.abs(y_mel_log.astype(mx.float32) - y_hat_mel_log)) * hps.train.c_mel
    loss_kl_log = kl_loss(z_p_log, logs_q_log, m_p_log, logs_p_log, z_mask_log) * hps.train.c_kl
    loss_fm_log = feature_loss(fmap_r, fmap_g)
    loss_gen_log = generator_loss(y_d_hat_g)
    mx.eval(loss_disc, loss_gen_log, loss_fm_log, loss_mel_log, loss_kl_log, loss_gen_all)
    return {
        "loss_disc": float(np.array(loss_disc)),
        "loss_gen": float(np.array(loss_gen_log)),
        "loss_fm": float(np.array(loss_fm_log)),
        "loss_mel": float(np.array(loss_mel_log)),
        "loss_kl": float(np.array(loss_kl_log)),
    }


def _to_mx_batch(batch, use_f0: bool, compute_dtype):
    if use_f0:
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _wave_lengths, sid = batch
        return (
            mx.array(phone, dtype=compute_dtype),
            mx.array(phone_lengths, dtype=mx.int32),
            mx.array(pitch, dtype=mx.int32),
            mx.array(pitchf, dtype=compute_dtype),
            mx.array(spec, dtype=compute_dtype),
            mx.array(spec_lengths, dtype=mx.int32),
            mx.array(wave, dtype=compute_dtype),
            mx.array(sid, dtype=mx.int32),
        )
    phone, phone_lengths, spec, spec_lengths, wave, _wave_lengths, sid = batch
    return (
        mx.array(phone, dtype=compute_dtype),
        mx.array(phone_lengths, dtype=mx.int32),
        mx.array(spec, dtype=compute_dtype),
        mx.array(spec_lengths, dtype=mx.int32),
        mx.array(wave, dtype=compute_dtype),
        mx.array(sid, dtype=mx.int32),
    )


def _resolve_precision(precision: str):
    if precision == "fp32":
        return mx.float32
    if precision == "bf16":
        return mx.bfloat16
    if precision == "fp16":
        return mx.float16
    raise ValueError("precision must be one of: fp32, bf16, fp16")


def _logger(exp_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"minirvc.mlx.train.{exp_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(exp_dir / "train_mlx.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
