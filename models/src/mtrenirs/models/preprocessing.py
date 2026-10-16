# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import numpy as np

from .recording import Recording


def normalize_run(signal: np.ndarray, clip_sigma: float = 6.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    mean = signal.mean(axis=1, keepdims=True)
    standard_deviation = signal.std(axis=1, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1.0
    normalized = (signal - mean) / standard_deviation
    return np.clip(normalized, -clip_sigma, clip_sigma).astype(np.float32)


def from_mne_raw(raw, *, align_positions: bool = True) -> Recording:
    try:
        import mne
    except ImportError as error:
        raise ImportError(
            'Install raw support with: pip install "mtrenirs-models[raw]"'
        ) from error

    processed = raw.copy().load_data()
    channel_types = processed.get_channel_types()
    if any(kind == "fnirs_cw_amplitude" for kind in channel_types):
        processed = mne.preprocessing.nirs.optical_density(processed)
        processed = mne.preprocessing.nirs.temporal_derivative_distribution_repair(processed)
        processed = mne.preprocessing.nirs.beer_lambert_law(processed, ppf=6.0)
    elif any(kind == "fnirs_od" for kind in channel_types):
        processed = mne.preprocessing.nirs.temporal_derivative_distribution_repair(processed)
        processed = mne.preprocessing.nirs.beer_lambert_law(processed, ppf=6.0)
    processed.filter(0.01, 0.5, verbose="ERROR")
    processed.resample(10.0, verbose="ERROR")
    positions = np.asarray([channel["loc"][:3] for channel in processed.info["chs"]])
    names = tuple(processed.ch_names)
    types = tuple(processed.get_channel_types())
    if align_positions:
        try:
            from mtrenirs.alignment import align
        except ImportError as error:
            raise ImportError(
                'Install alignment support with: pip install "mtrenirs-models[raw]"'
            ) from error
        mapping = align(positions=positions, names=names, unit="m", modality="fnirs")
        positions = mapping.ch_pos_1005
        valid = mapping.valid
    else:
        valid = np.isfinite(positions).all(axis=1)
    return Recording(
        signal=normalize_run(processed.get_data()),
        sfreq=10.0,
        ch_names=names,
        ch_types=types,
        ch_pos_1005=positions,
        ch_pos_1005_valid=valid,
        normalized=True,
    )


def load_snirf(path: str, *, align_positions: bool = True) -> Recording:
    try:
        import mne
    except ImportError as error:
        raise ImportError(
            'Install raw support with: pip install "mtrenirs-models[raw]"'
        ) from error
    raw = mne.io.read_raw_snirf(path, preload=True, verbose="ERROR")
    return from_mne_raw(raw, align_positions=align_positions)
