from __future__ import annotations


def resolve_device(device: str | None = None):
    import mlx.core as mx

    if device is None or device in {"auto", "mlx", "gpu"}:
        selected = mx.gpu if mx.device_count(mx.gpu) else mx.cpu
    elif device == "cpu":
        selected = mx.cpu
    else:
        raise ValueError("MLX backend only accepts device values: auto, mlx, gpu, cpu")
    mx.set_default_device(selected)
    return selected


def synchronize() -> None:
    import mlx.core as mx

    mx.eval()

