from __future__ import annotations

import queue
import threading
import time

import numpy as np

from minirvc.realtime.engine import RealtimeEngine


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def run_stream(
    engine: RealtimeEngine,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    channels: int = 1,
    queue_size: int = 4,
) -> None:
    import sounddevice as sd

    in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
    out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    error: list[BaseException] = []

    def push_latest(q: queue.Queue[np.ndarray], item: np.ndarray) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)

    def worker() -> None:
        while not stop.is_set():
            try:
                block = in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                push_latest(out_q, engine.process_block(block))
            except BaseException as exc:
                error.append(exc)
                stop.set()
                return

    def callback(indata, outdata, frames, _time, _status) -> None:
        if frames != engine.block_frame:
            raise RuntimeError(f"expected {engine.block_frame} frames, got {frames}")
        mono = np.asarray(indata, dtype=np.float32).mean(axis=1)
        push_latest(in_q, mono.copy())
        try:
            block = out_q.get_nowait()
        except queue.Empty:
            block = np.zeros(engine.block_frame, dtype=np.float32)
        if channels == 1:
            outdata[:, 0] = block
        else:
            outdata[:] = np.repeat(block[:, None], channels, axis=1)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        with sd.Stream(
            samplerate=engine.sample_rate,
            blocksize=engine.block_frame,
            channels=channels,
            dtype="float32",
            input_device=input_device,
            output_device=output_device,
            callback=callback,
        ):
            while not stop.is_set():
                if error:
                    raise error[0]
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join(timeout=1.0)
