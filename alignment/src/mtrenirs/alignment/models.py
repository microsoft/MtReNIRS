from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OptodeGeometry:
    source_names: tuple[str, ...]
    source_positions: np.ndarray
    detector_names: tuple[str, ...]
    detector_positions: np.ndarray
    channel_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        sources = np.asarray(self.source_positions, dtype=np.float64)
        detectors = np.asarray(self.detector_positions, dtype=np.float64)
        if sources.ndim != 2 or detectors.ndim != 2 or sources.shape[1:] != detectors.shape[1:]:
            raise ValueError("source and detector positions must be 2D arrays with matching dimensions")
        if len(self.source_names) != len(sources) or len(self.detector_names) != len(detectors):
            raise ValueError("optode names and positions must have matching lengths")
        if any(source < 0 or source >= len(sources) or detector < 0 or detector >= len(detectors)
               for source, detector in self.channel_pairs):
            raise ValueError("channel_pairs contains an invalid optode index")
        object.__setattr__(self, "source_positions", sources)
        object.__setattr__(self, "detector_positions", detectors)


@dataclass(frozen=True)
class MontageInput:
    names: tuple[str, ...]
    positions: np.ndarray
    unit: str = "m"
    frame: str = "head"
    modality: str = "fnirs"
    optodes: OptodeGeometry | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("positions must have shape (channels, 2) or (channels, 3)")
        if len(self.names) != len(positions):
            raise ValueError("names and positions must have the same length")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class AlignmentResult:
    names: tuple[str, ...]
    source_positions: np.ndarray
    slot_ids: np.ndarray
    slot_labels: tuple[str, ...]
    ch_pos_1005: np.ndarray
    valid: np.ndarray
    distances: np.ndarray
    distance_unit: str
    collisions: dict[int, tuple[str, ...]] = field(default_factory=dict)
    method: str = "nearest"
    reference: str = "standard_1005"

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ch_names=np.asarray(self.names),
            ch_slot_1005_id=self.slot_ids,
            ch_slot_1005_label=np.asarray(self.slot_labels),
            ch_pos_1005=self.ch_pos_1005.astype(np.float32),
            ch_pos_1005_valid=self.valid,
            ch_pos_1005_distance=self.distances.astype(np.float32),
            meta_ch_pos_1005_distance_unit=np.asarray(self.distance_unit),
            meta_ch_pos_1005_frame=np.asarray(self.reference),
            meta_alignment_method=np.asarray(self.method),
        )
        return path

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def json_safe(value):
            if isinstance(value, np.ndarray):
                return json_safe(value.tolist())
            if isinstance(value, list):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        payload = {
            "ch_names": list(self.names),
            "ch_slot_1005_id": self.slot_ids.tolist(),
            "ch_slot_1005_label": list(self.slot_labels),
            "ch_pos_1005": json_safe(self.ch_pos_1005),
            "ch_pos_1005_valid": self.valid.tolist(),
            "ch_pos_1005_distance": json_safe(self.distances),
            "collisions": {str(slot): list(names) for slot, names in self.collisions.items()},
            "meta_ch_pos_1005_distance_unit": self.distance_unit,
            "meta_ch_pos_1005_frame": self.reference,
            "meta_alignment_method": self.method,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def save_tsv(self, path: str | Path) -> Path:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                ["name", "slot_id", "slot_label", "x", "y", "z", "valid", "distance", "distance_unit"]
            )
            for index, name in enumerate(self.names):
                writer.writerow(
                    [
                        name,
                        int(self.slot_ids[index]),
                        self.slot_labels[index],
                        *self.ch_pos_1005[index].tolist(),
                        bool(self.valid[index]),
                        float(self.distances[index]),
                        self.distance_unit,
                    ]
                )
        return path

    def plot(self, path: str | Path, *, title: str | None = None) -> Path:
        from .plotting import plot_alignment

        return plot_alignment(self, path, title=title)
