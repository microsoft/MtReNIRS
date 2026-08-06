from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Recording:
    signal: np.ndarray
    sfreq: float
    ch_names: tuple[str, ...]
    ch_types: tuple[str, ...]
    ch_pos_1005: np.ndarray
    ch_pos_1005_valid: np.ndarray | None = None
    normalized: bool = False

    def __post_init__(self) -> None:
        signal = np.asarray(self.signal, dtype=np.float32)
        positions = np.asarray(self.ch_pos_1005, dtype=np.float32)
        if signal.ndim != 2:
            raise ValueError("signal must have shape (channels, samples)")
        if len(self.ch_names) != signal.shape[0] or len(self.ch_types) != signal.shape[0]:
            raise ValueError("channel metadata must match the signal channel axis")
        if positions.shape != (signal.shape[0], 3):
            raise ValueError("ch_pos_1005 must have shape (channels, 3)")
        valid = self.ch_pos_1005_valid
        if valid is None:
            valid = np.isfinite(positions).all(axis=1)
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (signal.shape[0],):
            raise ValueError("ch_pos_1005_valid must have shape (channels,)")
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "ch_pos_1005", positions)
        object.__setattr__(self, "ch_pos_1005_valid", valid)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Recording":
        with np.load(path, allow_pickle=False) as data:
            required = {
                "signal",
                "sfreq",
                "ch_names",
                "ch_types",
                "ch_pos_1005",
                "normalized",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"Canonical recording NPZ is missing: {', '.join(missing)}")
            valid = (
                np.asarray(data["ch_pos_1005_valid"])
                if "ch_pos_1005_valid" in data
                else None
            )
            return cls(
                signal=np.asarray(data["signal"]),
                sfreq=float(np.asarray(data["sfreq"])),
                ch_names=tuple(np.asarray(data["ch_names"]).astype(str)),
                ch_types=tuple(np.asarray(data["ch_types"]).astype(str)),
                ch_pos_1005=np.asarray(data["ch_pos_1005"]),
                ch_pos_1005_valid=valid,
                normalized=bool(np.asarray(data["normalized"])),
            )

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signal=self.signal,
            sfreq=np.asarray(self.sfreq),
            ch_names=np.asarray(self.ch_names),
            ch_types=np.asarray(self.ch_types),
            ch_pos_1005=self.ch_pos_1005,
            ch_pos_1005_valid=self.ch_pos_1005_valid,
            normalized=np.asarray(self.normalized),
        )
        return path
