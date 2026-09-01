# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import numpy as np


def minimum_overlap_starts(length: int, window: int = 100) -> np.ndarray:
    if length < window:
        raise ValueError("segment is shorter than one model window")
    count = int(np.ceil(length / window))
    if count == 1:
        return np.asarray([0], dtype=np.int64)
    return np.rint(np.linspace(0, length - window, count)).astype(np.int64)


def window_starts(length: int, window: int = 100, remainder: str = "drop") -> np.ndarray:
    if length < window:
        if remainder == "pad":
            return np.asarray([0], dtype=np.int64)
        raise ValueError("segment is shorter than one model window")
    if remainder == "drop":
        return np.arange(0, length - window + 1, window, dtype=np.int64)
    if remainder == "pad":
        return np.arange(0, length, window, dtype=np.int64)
    if remainder == "overlap":
        return minimum_overlap_starts(length, window)
    raise ValueError("remainder must be drop, pad, or overlap")


def extract_windows(
    signal: np.ndarray,
    *,
    onset: int,
    length: int,
    window: int = 100,
    short_policy: str = "error",
    remainder: str = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal)
    if onset < 0 or length <= 0 or onset + length > signal.shape[-1]:
        raise ValueError("requested segment is outside the recording")
    if length < window:
        if short_policy == "error":
            raise ValueError("segment is shorter than 10 seconds; use context or pad")
        if short_policy == "context":
            start = min(max(0, onset - (window - length) // 2), signal.shape[-1] - window)
            if start < 0 or start + window > signal.shape[-1]:
                raise ValueError("recording is too short to provide a context window")
            return signal[:, start : start + window][None], np.asarray([start])
        if short_policy == "pad":
            output = np.zeros((1, signal.shape[0], window), dtype=signal.dtype)
            output[0, :, :length] = signal[:, onset : onset + length]
            return output, np.asarray([onset])
        raise ValueError("short_policy must be error, context, or pad")
    relative = window_starts(length, window, remainder)
    absolute = onset + relative
    if remainder == "pad" and absolute[-1] + window > onset + length:
        windows = np.zeros(
            (len(absolute), signal.shape[0], window), dtype=signal.dtype
        )
        segment_end = onset + length
        for index, start in enumerate(absolute):
            available = min(window, segment_end - start)
            windows[index, :, :available] = signal[:, start : start + available]
    else:
        windows = np.stack([signal[:, start : start + window] for start in absolute])
    return windows, absolute
