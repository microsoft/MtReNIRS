# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np

from .models import MontageInput, OptodeGeometry


def load_positions_tsv(
    path: str | Path,
    *,
    unit: str = "m",
    frame: str = "head",
    modality: str = "fnirs",
) -> MontageInput:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("TSV contains no positions")
    required = {"name", "x", "y"}
    if not required.issubset(rows[0]):
        raise ValueError("TSV must contain name, x, and y columns")
    has_z = "z" in rows[0] and all(row.get("z", "") not in ("", "n/a") for row in rows)
    axes = ("x", "y", "z") if has_z else ("x", "y")
    positions = np.asarray([[float(row[axis]) for axis in axes] for row in rows])
    return MontageInput(tuple(row["name"] for row in rows), positions, unit, frame, modality)


def load_npz(path: str | Path) -> MontageInput:
    with np.load(path, allow_pickle=False) as data:
        position_key = "ch_pos" if "ch_pos" in data else "ch_pos_1005"
        if "ch_names" not in data or position_key not in data:
            raise ValueError("NPZ must contain ch_names and ch_pos or ch_pos_1005")
        unit = str(data["meta_ch_pos_unit"]) if "meta_ch_pos_unit" in data else "m"
        if "meta_ch_pos_frame" in data:
            frame = str(data["meta_ch_pos_frame"])
        elif "meta_ch_pos_1005_frame" in data:
            frame = str(data["meta_ch_pos_1005_frame"])
        else:
            frame = "head"
        return MontageInput(
            tuple(np.asarray(data["ch_names"]).astype(str)),
            np.asarray(data[position_key]),
            unit,
            frame,
        )


def load_json(path: str | Path) -> MontageInput:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    position_key = "ch_pos" if "ch_pos" in payload else "ch_pos_1005"
    if "ch_names" not in payload or position_key not in payload:
        raise ValueError("JSON must contain ch_names and ch_pos or ch_pos_1005")
    unit = str(payload.get("meta_ch_pos_unit", "m"))
    frame = str(
        payload.get("meta_ch_pos_frame", payload.get("meta_ch_pos_1005_frame", "head"))
    )
    positions = np.asarray(
        [
            [np.nan if coordinate is None else coordinate for coordinate in row]
            for row in payload[position_key]
        ],
        dtype=np.float64,
    )
    return MontageInput(tuple(map(str, payload["ch_names"])), positions, unit, frame)


def from_mne_raw(
    raw,
    *,
    frame: str = "head",
    modality: str = "fnirs",
    physical_pairs: bool = False,
    min_sd_distance_mm: float | None = None,
    max_sd_distance_mm: float | None = None,
) -> MontageInput:
    positions = np.asarray(
        [channel["loc"][:3] for channel in raw.info["chs"]],
        dtype=np.float64,
    )
    names = tuple(raw.ch_names)
    optodes = None
    if physical_pairs:
        selected: list[int] = []
        seen: set[str] = set()
        for index, (name, channel) in enumerate(zip(names, raw.info["chs"])):
            pair = _channel_source_detector(name)
            pair_name = f"{pair[0]}-{pair[1]}" if pair is not None else name
            if pair_name in seen:
                continue
            loc = np.asarray(channel["loc"], dtype=np.float64)
            if loc.size >= 9 and np.isfinite(loc[3:9]).all():
                distance_mm = float(np.linalg.norm(loc[3:6] - loc[6:9]) * 1e3)
                if min_sd_distance_mm is not None and distance_mm < min_sd_distance_mm:
                    continue
                if max_sd_distance_mm is not None and distance_mm > max_sd_distance_mm:
                    continue
            seen.add(pair_name)
            selected.append(index)
        positions = positions[selected]
        names = tuple(
            f"{pair[0]}-{pair[1]}" if (pair := _channel_source_detector(names[index])) else names[index]
            for index in selected
        )
        source_names: list[str] = []
        detector_names: list[str] = []
        source_positions: list[np.ndarray] = []
        detector_positions: list[np.ndarray] = []
        channel_pairs: list[tuple[int, int]] = []
        for index in selected:
            pair = _channel_source_detector(raw.ch_names[index])
            loc = np.asarray(raw.info["chs"][index]["loc"], dtype=np.float64)
            if pair is None or loc.size < 9 or not np.isfinite(loc[3:9]).all():
                channel_pairs = []
                break
            if pair[0] not in source_names:
                source_names.append(pair[0])
                source_positions.append(loc[3:6])
            if pair[1] not in detector_names:
                detector_names.append(pair[1])
                detector_positions.append(loc[6:9])
            channel_pairs.append((source_names.index(pair[0]), detector_names.index(pair[1])))
        if channel_pairs:
            optodes = OptodeGeometry(
                tuple(source_names),
                np.asarray(source_positions),
                tuple(detector_names),
                np.asarray(detector_positions),
                tuple(channel_pairs),
            )
    finite = np.isfinite(positions).all(axis=1)
    if finite.sum() < 2 or np.unique(positions[finite], axis=0).shape[0] < 2:
        raise ValueError(
            "SNIRF does not contain at least two distinct finite channel positions; "
            "provide a geometry sidecar or use montage-align-ui"
        )
    return MontageInput(names, positions, "m", frame, modality, optodes)


def load_snirf(
    path: str | Path,
    *,
    physical_pairs: bool = False,
    min_sd_distance_mm: float | None = None,
    max_sd_distance_mm: float | None = None,
) -> MontageInput:
    import mne

    raw = mne.io.read_raw_snirf(path, preload=False, verbose="ERROR")
    return from_mne_raw(
        raw,
        physical_pairs=physical_pairs,
        min_sd_distance_mm=min_sd_distance_mm,
        max_sd_distance_mm=max_sd_distance_mm,
    )


def load_position_text(
    path: str | Path,
    *,
    unit: str = "m",
    frame: str = "head",
) -> MontageInput:
    path = Path(path)
    sample = path.read_text(encoding="utf-8-sig")
    nonempty_lines = [line for line in sample.splitlines() if line.strip()]
    coordinate_pattern = re.compile(
        r"^\s*([^,]+?)(?:,|(?=-\d))\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
    )
    coordinate_matches = [coordinate_pattern.match(line) for line in nonempty_lines]
    if len(coordinate_matches) >= 2 and all(coordinate_matches):
        return MontageInput(
            tuple(match.group(1).strip() for match in coordinate_matches),
            np.asarray(
                [[float(match.group(axis)) for axis in range(2, 5)] for match in coordinate_matches],
                dtype=np.float64,
            ),
            unit,
            frame,
        )

    headerless_rows: list[list[str]] = []
    for line in nonempty_lines:
        values = [value.strip() for value in re.split(r"[\t,;]", line)]
        if len(values) != 4:
            headerless_rows = []
            break
        try:
            [float(value) for value in values[1:]]
        except ValueError:
            headerless_rows = []
            break
        headerless_rows.append(values)
    if len(headerless_rows) >= 2:
        return MontageInput(
            tuple(row[0] for row in headerless_rows),
            np.asarray([row[1:] for row in headerless_rows], dtype=np.float64),
            unit,
            frame,
        )

    try:
        dialect = csv.Sniffer().sniff(sample[:4096], delimiters="\t,;")
        rows = list(csv.DictReader(sample.splitlines(), dialect=dialect))
    except csv.Error:
        lines = [line.split() for line in sample.splitlines() if line.strip()]
        rows = [dict(zip(lines[0], values)) for values in lines[1:]] if lines else []
    if not rows:
        raise ValueError(f"{path.name} contains no position rows")

    columns = {column.lower().strip(): column for column in rows[0]}

    def find_column(*aliases: str) -> str | None:
        return next((columns[alias] for alias in aliases if alias in columns), None)

    name_column = find_column("name", "channel", "ch_name", "label", "optode")
    axis_columns = [find_column(axis, f"pos_{axis}", f"{axis}_m", f"{axis}_mm") for axis in "xyz"]
    if name_column is None or axis_columns[0] is None or axis_columns[1] is None:
        raise ValueError(
            f"{path.name} needs a name/channel/label column and x/y columns; "
            "rename non-standard columns in the source table"
        )
    selected_axes = axis_columns if axis_columns[2] is not None else axis_columns[:2]
    positions = np.asarray(
        [[float(row[column]) for column in selected_axes] for row in rows],
        dtype=np.float64,
    )
    return MontageInput(
        tuple(str(row[name_column]).strip() for row in rows),
        positions,
        unit,
        frame,
    )


def _channel_source_detector(name: str) -> tuple[str, str] | None:
    match = re.search(r"(?i)(S\d+)\s*[_-]?\s*(D\d+)", name)
    return (match.group(1).upper(), match.group(2).upper()) if match else None


def load_bids_nirs_sidecars(
    optodes_path: str | Path,
    coordsystem_path: str | Path,
    channels_path: str | Path,
    *,
    min_sd_distance_mm: float | None = None,
    max_sd_distance_mm: float | None = None,
) -> MontageInput:
    metadata = json.loads(Path(coordsystem_path).read_text(encoding="utf-8"))
    unit = str(metadata.get("NIRSCoordinateUnits", metadata.get("EEGCoordinateUnits", "m")))
    with Path(optodes_path).open(encoding="utf-8-sig", newline="") as handle:
        optode_rows = list(csv.DictReader(handle, delimiter="\t"))
    with Path(channels_path).open(encoding="utf-8-sig", newline="") as handle:
        channel_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not optode_rows or not channel_rows or "name" not in channel_rows[0]:
        raise ValueError("BIDS NIRS import needs non-empty optodes.tsv and channels.tsv with name")
    optodes = {
        str(row["name"]).upper(): np.asarray([row[axis] for axis in ("x", "y", "z")], dtype=np.float64)
        for row in optode_rows
        if all(row.get(axis, "").lower() not in {"", "n/a", "nan"} for axis in ("x", "y", "z"))
    }
    names: list[str] = []
    positions: list[np.ndarray] = []
    channel_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in channel_rows:
        channel_type = str(row.get("type", "")).upper()
        if channel_type and not channel_type.startswith("NIRS"):
            continue
        source = str(row.get("source", "")).upper()
        detector = str(row.get("detector", "")).upper()
        pair = (source, detector) if source.startswith("S") and detector.startswith("D") else None
        if pair is None:
            pair = _channel_source_detector(str(row["name"]))
        if pair is None or pair[0] not in optodes or pair[1] not in optodes:
            continue
        if pair in seen_pairs:
            continue
        distance_mm = float(np.linalg.norm(optodes[pair[0]] - optodes[pair[1]]))
        unit_scale_mm = {"m": 1e3, "meter": 1e3, "metre": 1e3, "cm": 10.0, "mm": 1.0}.get(
            unit.lower()
        )
        if unit_scale_mm is None:
            raise ValueError(f"Unsupported BIDS NIRS coordinate unit: {unit}")
        distance_mm *= unit_scale_mm
        if min_sd_distance_mm is not None and distance_mm < min_sd_distance_mm:
            continue
        if max_sd_distance_mm is not None and distance_mm > max_sd_distance_mm:
            continue
        seen_pairs.add(pair)
        names.append(f"{pair[0]}-{pair[1]}")
        positions.append((optodes[pair[0]] + optodes[pair[1]]) / 2.0)
        channel_pairs.append((pair[0], pair[1]))
    if not positions:
        raise ValueError("No channels.tsv names matched S<number>-D<number> optodes")
    source_names = tuple(dict.fromkeys(source for source, _ in channel_pairs))
    detector_names = tuple(dict.fromkeys(detector for _, detector in channel_pairs))
    optode_geometry = OptodeGeometry(
        source_names,
        np.asarray([optodes[name] for name in source_names]),
        detector_names,
        np.asarray([optodes[name] for name in detector_names]),
        tuple((source_names.index(source), detector_names.index(detector)) for source, detector in channel_pairs),
    )
    return MontageInput(tuple(names), np.asarray(positions), unit, "head", "fnirs", optode_geometry)


def _xdf_scalar(value) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    return str(value)


def _muse_s_athena_montage() -> MontageInput:
    import mne

    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    fpz, fp1, fp2 = (np.asarray(positions[name], dtype=np.float64) for name in ("Fpz", "Fp1", "Fp2"))
    left = (fp1 - fpz) / np.linalg.norm(fp1 - fpz)
    right = (fp2 - fpz) / np.linalg.norm(fp2 - fpz)
    detectors = (
        fpz + left * 0.029,
        fpz + right * 0.029,
        fpz + left * 0.008,
        fpz + right * 0.008,
    )
    return MontageInput(
        ("S1-D1", "S1-D2", "S1-D3", "S1-D4"),
        np.asarray([(fpz + detector) / 2.0 for detector in detectors]),
        "m",
        "standard_1005",
        "fnirs",
        OptodeGeometry(
            ("S1",),
            np.asarray([fpz]),
            ("D1", "D2", "D3", "D4"),
            np.asarray(detectors),
            ((0, 0), (0, 1), (0, 2), (0, 3)),
        ),
    )


def _is_muse_s_athena_optics(info: dict) -> bool:
    name = _xdf_scalar(info.get("name", [""])).upper()
    stream_type = _xdf_scalar(info.get("type", [""])).upper()
    try:
        channel_count = int(_xdf_scalar(info.get("channel_count", ["0"])))
    except ValueError:
        return False
    return name.startswith("MUSE-OPTICS") and stream_type == "PPG" and channel_count == 16


def load_xdf(path: str | Path) -> MontageInput:
    try:
        import pyxdf
    except ImportError as error:
        raise ImportError("XDF import requires the pyxdf package") from error

    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=False)
    candidates: list[MontageInput] = []
    for stream in streams:
        info = stream.get("info", {})
        if _is_muse_s_athena_optics(info):
            candidates.append(_muse_s_athena_montage())
            continue
        try:
            channels = info["desc"][0]["channels"][0]["channel"]
        except (KeyError, IndexError, TypeError):
            continue
        names: list[str] = []
        positions: list[list[float]] = []
        for channel in channels:
            location = channel.get("location", [{}])[0] if isinstance(channel, dict) else {}
            coordinate_source = location if isinstance(location, dict) else channel
            try:
                xyz = [float(_xdf_scalar(coordinate_source[axis])) for axis in ("x", "y", "z")]
            except (KeyError, TypeError, ValueError):
                continue
            names.append(_xdf_scalar(channel.get("label", [f"ch{len(names) + 1}"])))
            positions.append(xyz)
        if len(positions) >= 2:
            unit = _xdf_scalar(info.get("coordinate_unit", ["m"]))
            candidates.append(MontageInput(tuple(names), np.asarray(positions), unit, "head"))
    if not candidates:
        raise ValueError(
            "XDF has no stream with per-channel x/y/z metadata; upload a position TSV/JSON sidecar"
        )
    return max(candidates, key=lambda montage: len(montage.names))
