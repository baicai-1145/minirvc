from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def load_audio(path: str | Path, sample_rate: int, highpass_hz: float = 48.0) -> np.ndarray:
    path = Path(path)
    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        filter_graph = None
        if highpass_hz > 0:
            low_q = 0.6180339887498948
            high_q = 1.618033988749895
            filter_graph = av.filter.Graph()
            filter_graph.link_nodes(
                filter_graph.add(
                    "abuffer",
                    f"time_base=1/{sample_rate}:sample_rate={sample_rate}:sample_fmt=fltp:channel_layout=mono",
                ),
                filter_graph.add("highpass", f"f={highpass_hz}:p=2:w={low_q}"),
                filter_graph.add("highpass", f"f={highpass_hz}:p=2:w={high_q}"),
                filter_graph.add("highpass", f"f={highpass_hz}:p=1"),
                filter_graph.add("abuffersink"),
            ).configure()

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                if filter_graph is None:
                    chunks.append(resampled.to_ndarray().reshape(-1))
                    continue
                _push_filter_output(filter_graph, resampled, chunks)

        for resampled in resampler.resample(None):
            if filter_graph is None:
                chunks.append(resampled.to_ndarray().reshape(-1))
                continue
            _push_filter_output(filter_graph, resampled, chunks)

        if filter_graph is not None:
            try:
                filter_graph.push(None)
            except Exception:
                pass
            _drain_filter_output(filter_graph, chunks)

    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def load_audio_unfiltered(path: str | Path, sample_rate: int) -> np.ndarray:
    return load_audio(path, sample_rate, highpass_hz=0.0)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)

    with av.open(str(path), mode="w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=sample_rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=dst_rate)
    frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = src_rate
    resampled_chunks: list[np.ndarray] = []
    for output in resampler.resample(frame):
        resampled_chunks.append(output.to_ndarray().reshape(-1))
    for output in resampler.resample(None):
        resampled_chunks.append(output.to_ndarray().reshape(-1))
    resampled = np.concatenate(resampled_chunks) if resampled_chunks else np.empty(0, dtype=np.float32)
    target_length = int(np.ceil(audio.shape[-1] * float(dst_rate) / src_rate))
    if resampled.shape[-1] > target_length:
        resampled = resampled[:target_length]
    elif resampled.shape[-1] < target_length:
        resampled = np.pad(resampled, (0, target_length - resampled.shape[-1]))
    return np.asarray(resampled, dtype=audio.dtype)


def _push_filter_output(graph: av.filter.Graph, frame: av.AudioFrame, chunks: list[np.ndarray]) -> None:
    graph.push(frame)
    _drain_filter_output(graph, chunks)


def _drain_filter_output(graph: av.filter.Graph, chunks: list[np.ndarray]) -> None:
    while True:
        try:
            output = graph.pull()
        except (av.BlockingIOError, av.EOFError):
            break
        chunks.append(output.to_ndarray().reshape(-1))
