# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .models import AlignmentResult, MontageInput

_UNIT_SCALE = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}


def load_reference(name: str = "standard_1005") -> dict[str, object]:
    if name not in {"standard_1005", "standard_1020"}:
        raise ValueError("reference must be 'standard_1005' or 'standard_1020'")
    import mne

    montage = mne.channels.make_standard_montage(name)
    positions = montage.get_positions()["ch_pos"]
    labels = tuple(montage.ch_names)
    xyz = np.asarray([positions[label] for label in labels], dtype=np.float64)
    return {"name": name, "labels": labels, "positions": xyz, "positions_2d": _to_2d(xyz)}


def _to_2d(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    output = np.full((len(positions), 2), np.nan)
    valid = np.isfinite(positions).all(axis=1)
    if not valid.any():
        return output
    points = positions[valid]
    radii = np.linalg.norm(points, axis=1, keepdims=True)
    radii[radii == 0] = 1.0
    unit = points / radii
    polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    output[valid, 0] = polar * np.cos(azimuth)
    output[valid, 1] = polar * np.sin(azimuth)
    return output


def _isotropic_align(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source).all(axis=1)
    if not valid.any():
        return source.copy()
    output = source.copy()
    source_center = source[valid].mean(axis=0)
    reference_center = reference.mean(axis=0)
    source_rms = float(np.sqrt(np.square(source[valid] - source_center).sum(axis=1).mean()))
    reference_rms = float(np.sqrt(np.square(reference - reference_center).sum(axis=1).mean()))
    if source_rms == 0:
        raise ValueError("at least two distinct positions are required for coordinate alignment")
    output[valid] = (source[valid] - source_center) / source_rms * reference_rms + reference_center
    return output


def transform_positions(
    montage: MontageInput,
    positions: np.ndarray,
    *,
    reference: str = "standard_1005",
) -> np.ndarray:
    """Apply the montage alignment transform to auxiliary positions such as optodes."""
    if montage.unit not in _UNIT_SCALE:
        raise ValueError("unit must be one of: m, cm, mm")
    source = montage.positions * _UNIT_SCALE[montage.unit]
    auxiliary = np.asarray(positions, dtype=np.float64) * _UNIT_SCALE[montage.unit]
    if auxiliary.ndim != 2 or auxiliary.shape[1] != source.shape[1]:
        raise ValueError("auxiliary positions must match the montage coordinate dimensions")
    ref = load_reference(reference)
    target = np.asarray(ref["positions"] if source.shape[1] == 3 else ref["positions_2d"])
    if montage.frame == reference and source.shape[1] == 3:
        return auxiliary
    valid = np.isfinite(source).all(axis=1)
    source_center = source[valid].mean(axis=0)
    reference_center = target.mean(axis=0)
    source_rms = float(np.sqrt(np.square(source[valid] - source_center).sum(axis=1).mean()))
    reference_rms = float(np.sqrt(np.square(target - reference_center).sum(axis=1).mean()))
    if source_rms == 0:
        raise ValueError("at least two distinct positions are required for coordinate alignment")
    return (auxiliary - source_center) / source_rms * reference_rms + reference_center


def _assign(points: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slot_ids = np.full(len(points), -1, dtype=np.int64)
    distances = np.full(len(points), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if valid.any():
        pairwise = np.linalg.norm(points[valid, None, :] - reference[None, :, :], axis=2)
        slot_ids[valid] = pairwise.argmin(axis=1)
        distances[valid] = pairwise.min(axis=1)
    return slot_ids, distances


def _physical_channel_id(name: str) -> str:
    parts = str(name).split()
    if len(parts) < 2:
        return str(name)
    suffix = parts[-1].lower()
    is_wavelength = suffix.isdigit() and len(suffix) >= 3
    if suffix in {"hbo", "hbr", "hbt"} or is_wavelength:
        return " ".join(parts[:-1])
    return str(name)


def _collision_map(slot_ids: np.ndarray, names: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    grouped: dict[int, set[str]] = defaultdict(set)
    for slot_id, name in zip(slot_ids, names):
        if slot_id >= 0:
            grouped[int(slot_id)].add(_physical_channel_id(name))
    return {
        slot: tuple(sorted(values))
        for slot, values in grouped.items()
        if len(values) > 1
    }


def align(
    montage: MontageInput | None = None,
    *,
    positions: np.ndarray | None = None,
    names: list[str] | tuple[str, ...] | None = None,
    unit: str = "m",
    frame: str = "head",
    modality: str = "fnirs",
    reference: str = "standard_1005",
    canonical_labels: list[str] | tuple[str, ...] | None = None,
) -> AlignmentResult:
    if montage is None:
        if positions is None or names is None:
            raise ValueError("provide MontageInput or both positions and names")
        montage = MontageInput(tuple(names), positions, unit, frame, modality)
    if montage.unit not in _UNIT_SCALE:
        raise ValueError("unit must be one of: m, cm, mm")
    ref = load_reference(reference)
    ref_labels = ref["labels"]
    ref_positions = np.asarray(ref["positions"])
    source = montage.positions * _UNIT_SCALE[montage.unit]

    if canonical_labels is not None:
        if len(canonical_labels) != len(montage.names):
            raise ValueError("canonical_labels and names must have the same length")
        lookup = {label.lower(): index for index, label in enumerate(ref_labels)}
        slot_ids = np.asarray([lookup.get(label.lower(), -1) for label in canonical_labels])
        distances = np.full(len(slot_ids), np.nan, dtype=np.float64)
        aligned_source = np.full((len(source), 3), np.nan)
        valid = slot_ids >= 0
        method = "canonical_labels"
    elif montage.frame == reference and source.shape[1] == 3:
        aligned_source = source.copy()
        slot_ids, distances = _assign(aligned_source, ref_positions)
        method = "canonical_3d_nearest"
    elif source.shape[1] == 3 and np.isfinite(source).all(axis=1).any():
        aligned_source = _isotropic_align(source, ref_positions)
        slot_ids, distances = _assign(aligned_source, ref_positions)
        method = "isotropic_3d_nearest"
    else:
        source_2d = source[:, :2]
        ref_2d = np.asarray(ref["positions_2d"])
        aligned_source = _isotropic_align(source_2d, ref_2d)
        slot_ids, distances = _assign(aligned_source, ref_2d)
        method = "isotropic_2d_nearest"

    valid = slot_ids >= 0
    canonical = np.full((len(slot_ids), 3), np.nan, dtype=np.float64)
    canonical[valid] = ref_positions[slot_ids[valid]]
    labels = tuple(ref_labels[index] if index >= 0 else "" for index in slot_ids)
    return AlignmentResult(
        names=montage.names,
        source_positions=aligned_source,
        slot_ids=slot_ids,
        slot_labels=labels,
        ch_pos_1005=canonical,
        valid=valid,
        distances=distances,
        distance_unit=(
            "n/a"
            if method == "canonical_labels"
            else "projection" if aligned_source.shape[1] == 2 else "m"
        ),
        collisions=_collision_map(slot_ids, montage.names),
        method=method,
        reference=reference,
    )


def align_optodes(
    *,
    source_positions: np.ndarray,
    detector_positions: np.ndarray,
    channel_pairs: list[tuple[int, int]],
    channel_names: list[str] | tuple[str, ...] | None = None,
    unit: str = "m",
    frame: str = "head",
    reference: str = "standard_1005",
) -> AlignmentResult:
    sources = np.asarray(source_positions, dtype=np.float64)
    detectors = np.asarray(detector_positions, dtype=np.float64)
    if sources.ndim != 2 or detectors.ndim != 2 or sources.shape[1] != detectors.shape[1]:
        raise ValueError("source and detector positions must be 2D arrays with matching dimensions")
    midpoints = np.asarray([(sources[s] + detectors[d]) / 2 for s, d in channel_pairs])
    if midpoints.shape[1] == 3:
        radius = np.median(np.linalg.norm(np.vstack([sources, detectors]), axis=1))
        norms = np.linalg.norm(midpoints, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        midpoints[nonzero] = midpoints[nonzero] / norms[nonzero] * radius
    names = tuple(channel_names or [f"S{s + 1}_D{d + 1}" for s, d in channel_pairs])
    return align(positions=midpoints, names=names, unit=unit, frame=frame, reference=reference)
