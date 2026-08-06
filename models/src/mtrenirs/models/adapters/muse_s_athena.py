from __future__ import annotations

from pathlib import Path

import numpy as np

from ..preprocessing import from_mne_raw
from ..recording import Recording


_WAVELENGTHS = (730.0, 850.0)
_OPTICS_PAIRS = (
    ("S1-D1", 0, 2),
    ("S1-D2", 1, 3),
    ("S1-D3", 4, 6),
    ("S1-D4", 5, 7),
)


def _load_optics(path: str | Path) -> tuple[np.ndarray, float]:
    try:
        import pyxdf
    except ImportError as error:
        raise ImportError(
            'Install Muse support with: pip install "mtrenirs-models[muse]"'
        ) from error

    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=False)
    candidates = [
        stream
        for stream in streams
        if str(stream.get("info", {}).get("name", [""])[0]).upper().startswith("MUSE-OPTICS")
        and str(stream.get("info", {}).get("type", [""])[0]).upper() == "PPG"
        and len(stream.get("time_stamps", [])) >= 2
    ]
    if not candidates:
        raise ValueError("XDF has no non-empty Muse-OPTICS PPG stream")
    stream = max(candidates, key=lambda candidate: len(candidate["time_stamps"]))
    time_series = np.asarray(stream["time_series"], dtype=np.float64)
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    if time_series.ndim != 2 or time_series.shape[0] != len(timestamps) or time_series.shape[1] < 8:
        raise ValueError("Muse-OPTICS stream must have samples by at least 8 channels")
    intensity = time_series[:, :8].T
    if not np.isfinite(intensity).all() or np.any(intensity <= 0):
        raise ValueError("Muse-OPTICS intensity channels must be finite and positive")
    duration = float(timestamps[-1] - timestamps[0])
    if not np.isfinite(duration) or duration <= 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Muse-OPTICS timestamps must be finite and strictly increasing")
    return intensity, float((len(timestamps) - 1) / duration)


def _to_mne_raw(path: str | Path):
    try:
        import mne
        from mtrenirs.alignment import load_xdf as load_xdf_montage
    except ImportError as error:
        raise ImportError(
            'Install Muse support with: pip install "mtrenirs-models[muse]"'
        ) from error

    intensity, sample_rate = _load_optics(path)
    montage = load_xdf_montage(path)
    if montage.names != tuple(pair[0] for pair in _OPTICS_PAIRS):
        raise ValueError("Muse XDF montage does not match the expected four source-detector pairs")
    standard_positions = mne.channels.make_standard_montage("standard_1005").get_positions()[
        "ch_pos"
    ]
    source = np.asarray(standard_positions["Fpz"], dtype=np.float64)
    names: list[str] = []
    rows: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    for pair_index, (pair_name, wavelength_0_index, wavelength_1_index) in enumerate(
        _OPTICS_PAIRS
    ):
        midpoint = montage.positions[pair_index]
        detector = 2.0 * midpoint - source
        for wavelength, raw_index in zip(
            _WAVELENGTHS, (wavelength_0_index, wavelength_1_index)
        ):
            names.append(f"{pair_name.replace('-', '_')} {int(wavelength)}")
            rows.append(intensity[raw_index])
            location = np.zeros(12, dtype=np.float64)
            location[:3] = midpoint
            location[3:6] = source
            location[6:9] = detector
            location[9] = wavelength
            locations.append(location)
    info = mne.create_info(names, sample_rate, ch_types="fnirs_cw_amplitude")
    for channel, location in zip(info["chs"], locations):
        channel["loc"] = location
    return mne.io.RawArray(np.vstack(rows), info, verbose="ERROR"), montage


def load_muse_s_athena_xdf(
    path: str | Path,
    *,
    align_positions: bool = True,
) -> Recording:
    """Load one Muse S Athena XDF recording into the canonical MtReNIRS schema."""
    raw, montage = _to_mne_raw(path)
    recording = from_mne_raw(raw, align_positions=False)
    if not align_positions:
        return recording

    from mtrenirs.alignment import align

    mapping = align(montage)
    pair_positions = {
        name.replace("-", "_"): (position, valid)
        for name, position, valid in zip(mapping.names, mapping.ch_pos_1005, mapping.valid)
    }
    positions: list[np.ndarray] = []
    valid: list[bool] = []
    for name in recording.ch_names:
        pair_name = name.split()[0]
        if pair_name not in pair_positions:
            raise ValueError(f"Processed Muse channel has an unknown source-detector pair: {name}")
        position, is_valid = pair_positions[pair_name]
        positions.append(position)
        valid.append(bool(is_valid))
    return Recording(
        signal=recording.signal,
        sfreq=recording.sfreq,
        ch_names=recording.ch_names,
        ch_types=recording.ch_types,
        ch_pos_1005=np.asarray(positions),
        ch_pos_1005_valid=np.asarray(valid),
        normalized=recording.normalized,
    )