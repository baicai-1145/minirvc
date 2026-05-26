from __future__ import annotations

import mlx.core as mx


def feature_loss(fmap_r, fmap_g):
    loss = mx.array(0.0)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + mx.mean(mx.abs(mx.stop_gradient(rl.astype(mx.float32)) - gl.astype(mx.float32)))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = mx.array(0.0)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.astype(mx.float32)
        dg = dg.astype(mx.float32)
        loss = loss + mx.mean((1 - dr) ** 2) + mx.mean(dg**2)
    return loss


def generator_loss(disc_outputs):
    loss = mx.array(0.0)
    for dg in disc_outputs:
        dg = dg.astype(mx.float32)
        loss = loss + mx.mean((1 - dg) ** 2)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.astype(mx.float32)
    logs_q = logs_q.astype(mx.float32)
    m_p = m_p.astype(mx.float32)
    logs_p = logs_p.astype(mx.float32)
    z_mask = z_mask.astype(mx.float32)
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * mx.exp(-2.0 * logs_p)
    return mx.sum(kl * z_mask) / mx.sum(z_mask)

