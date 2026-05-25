from __future__ import annotations

import datetime as dt
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minirvc.train import checkpoint
from minirvc.train.config import load_config, to_namespace
from minirvc.train.dataset import BucketBatchSampler, RvcCollate, RvcDataset
from minirvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from minirvc.train.mel import mel_spectrogram_torch, spec_to_mel_torch
from minirvc.train.nn import commons
from minirvc.train.nn.models import (
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


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
    num_workers: int = 4,
    resume: bool = True,
    export_final: bool = True,
) -> None:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(exp_dir)
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
    hps.total_epoch = epochs
    hps.save_every_epoch = save_every_epoch

    device_obj = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    fp16 = bool(hps.train.fp16_run and device_obj.type == "cuda")
    _seed(int(hps.train.seed))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    dataset = RvcDataset(filelist, hps.data, use_f0)
    sampler = BucketBatchSampler(dataset.lengths, batch_size=batch_size, boundaries=[100, 200, 300, 400, 500, 600, 700, 800, 900])
    loader_kwargs = {
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": device_obj.type == "cuda",
        "collate_fn": RvcCollate(use_f0),
        "batch_sampler": sampler,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 8
    train_loader = DataLoader(dataset, **loader_kwargs)

    net_g = _build_generator(hps, use_f0, version, sample_rate, fp16).to(device_obj)
    net_d = _build_discriminator(hps, version).to(device_obj)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    start_epoch = 1
    global_step = 0
    if resume:
        latest_g = checkpoint.latest_checkpoint_path(exp_dir, "G_*.pth")
        latest_d = checkpoint.latest_checkpoint_path(exp_dir, "D_*.pth")
    else:
        latest_g = latest_d = None
    if latest_g is not None and latest_d is not None:
        _, d_epoch = checkpoint.load_training_checkpoint(latest_d, net_d, optim_d)
        _, g_epoch = checkpoint.load_training_checkpoint(latest_g, net_g, optim_g)
        start_epoch = min(g_epoch, d_epoch) + 1
        global_step = (start_epoch - 2) * len(train_loader)
        logger.info("resumed from epoch %s", start_epoch - 1)
    else:
        checkpoint.load_pretrained(pretrain_g, net_g)
        checkpoint.load_pretrained(pretrain_d, net_d)
        logger.info("loaded pretrained %s", pretrain_g)
        logger.info("loaded pretrained %s", pretrain_d)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=start_epoch - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=start_epoch - 2)
    scaler = torch.amp.GradScaler("cuda" if device_obj.type == "cuda" else "cpu", enabled=fp16)

    for epoch in range(start_epoch, epochs + 1):
        global_step = _train_epoch(
            epoch=epoch,
            hps=hps,
            use_f0=use_f0,
            net_g=net_g,
            net_d=net_d,
            optim_g=optim_g,
            optim_d=optim_d,
            scaler=scaler,
            train_loader=train_loader,
            device=device_obj,
            fp16=fp16,
            global_step=global_step,
            logger=logger,
        )
        scheduler_g.step()
        scheduler_d.step()
        if epoch % save_every_epoch == 0:
            checkpoint.save_training_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, exp_dir / f"G_{global_step}.pth")
            checkpoint.save_training_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, exp_dir / f"D_{global_step}.pth")
        logger.info("====> Epoch: %s [%s]", epoch, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    logger.info("Training is done. The program is closed.")
    if export_final:
        checkpoint.export_small_model(net_g, exp_dir / f"{exp_dir.name}.pth", hps, sample_rate, use_f0, version, epochs)
        logger.info("saving final ckpt:Success.")


def _train_epoch(
    *,
    epoch: int,
    hps,
    use_f0: bool,
    net_g: torch.nn.Module,
    net_d: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_loader: DataLoader,
    device: torch.device,
    fp16: bool,
    global_step: int,
    logger: logging.Logger,
) -> int:
    train_loader.batch_sampler.set_epoch(epoch)
    net_g.train()
    net_d.train()
    epoch_start = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        if use_f0:
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = _to_device(batch, device)
        else:
            phone, phone_lengths, spec, spec_lengths, wave, _, sid = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=fp16):
            if use_f0:
                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
            else:
                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(
                    phone, phone_lengths, spec, spec_lengths, sid
                )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if fp16:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with torch.amp.autocast(device_type=device.type, enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with torch.amp.autocast(device_type=device.type, enabled=fp16):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps.train.log_interval == 0:
            logger.info("Train Epoch: %s [%.0f%%]", epoch, 100.0 * batch_idx / len(train_loader))
            logger.info([global_step, optim_g.param_groups[0]["lr"]])
            logger.info(
                "loss_disc=%.3f, loss_gen=%.3f, loss_fm=%.3f,loss_mel=%.3f, loss_kl=%.3f",
                _scalar(loss_disc),
                _scalar(loss_gen),
                _scalar(loss_fm),
                _scalar(torch.clamp(loss_mel, max=75)),
                _scalar(torch.clamp(loss_kl, max=9)),
            )
            del losses_gen, losses_disc_r, losses_disc_g, grad_norm_d, grad_norm_g
        global_step += 1
    logger.info("epoch_time=%.3fs", time.perf_counter() - epoch_start)
    return global_step


def _build_generator(hps, use_f0: bool, version: str, sample_rate: str, fp16: bool) -> torch.nn.Module:
    cls = {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }[(version, use_f0)]
    args = (hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length)
    if use_f0:
        return cls(*args, **vars(hps.model), is_half=fp16, sr=sample_rate)
    return cls(*args, **vars(hps.model), is_half=fp16)


def _build_discriminator(hps, version: str) -> torch.nn.Module:
    cls = MultiPeriodDiscriminator if version == "v1" else MultiPeriodDiscriminatorV2
    return cls(hps.model.use_spectral_norm)


def _to_device(batch, device: torch.device):
    return tuple(item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _logger(exp_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"minirvc.train.{exp_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file = logging.FileHandler(exp_dir / "train.log", encoding="utf-8")
    file.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file)
    logger.propagate = False
    return logger
