# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import csv
from functools import lru_cache
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

import numpy as np

from mtrenirs.alignment.alignment import align, align_optodes, load_reference, transform_positions
from mtrenirs.alignment.io import (
    load_bids_nirs_sidecars,
    load_json,
    load_npz,
    load_position_text,
    load_positions_tsv,
    load_snirf,
    load_xdf,
)
from mtrenirs.alignment.models import AlignmentResult, MontageInput, OptodeGeometry

_CLASSIC_1020_LABELS = (
    "Fp1", "Fpz", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
)

_DEFAULT_SD_MIN_MM = 8.0
_DEFAULT_SD_MAX_MM = 35.0

_HEAD_MAP_HTML = '<div id="head-map-root"></div>'
_HEAD_MAP_CSS = """
#head-map-root { width: 100%; }
.head-map { width: 100%; height: min(78vh, 820px); min-height: 620px; display: block; }
.head-outline { fill: #f8faf9; stroke: #596963; stroke-width: 0.008; }
.landmark-dot { fill: #40534c; pointer-events: none; }
.landmark-label { fill: #26332f; font: 600 0.036px sans-serif; text-anchor: middle; pointer-events: none; }
.slot-dot { fill: #b8c2bd; stroke: white; stroke-width: 0.002; pointer-events: none; }
.slot-hit { fill: transparent; cursor: pointer; }
.slot:hover .slot-dot { fill: #6f827b; stroke: #26332f; stroke-width: 0.004; }
.source { fill: #d62728; pointer-events: none; }
.detector { fill: #1f77b4; pointer-events: none; }
.optode-label { fill: #111; font: 600 0.038px sans-serif; text-anchor: middle; pointer-events: none; }
.channel-line { stroke: #111; stroke-width: 0.008; pointer-events: none; }
.midpoint { fill: #ffd400; stroke: #111; stroke-width: 0.004; pointer-events: none; }
"""
_HEAD_MAP_JS = """
export default function(component) {
    const { data, setTriggerValue, parentElement } = component;
    const root = parentElement.querySelector('#head-map-root');
    root.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg';
    const make = (tag, attrs = {}) => {
        const node = document.createElementNS(ns, tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
    };
    const svg = make('svg', {viewBox: '-1.25 -1.28 2.5 2.56', class: 'head-map', role: 'img'});
    root.appendChild(svg);

    svg.appendChild(make('circle', {cx: 0, cy: 0, r: 1, class: 'head-outline'}));
    svg.appendChild(make('path', {d: 'M -0.13 -0.98 L 0 -1.16 L 0.13 -0.98', class: 'head-outline'}));
    svg.appendChild(make('path', {d: 'M -0.99 -0.22 Q -1.13 0 -0.99 0.22', class: 'head-outline'}));
    svg.appendChild(make('path', {d: 'M 0.99 -0.22 Q 1.13 0 0.99 0.22', class: 'head-outline'}));

    const occupiedLabels = new Set(
        [...data.sources, ...data.detectors].map(point => point.label)
    );
    data.landmarks.forEach(point => {
        svg.appendChild(make('circle', {cx: point.x, cy: point.y, r: 0.026, class: 'landmark-dot'}));
        if (!occupiedLabels.has(point.label)) {
            const label = make('text', {x: point.x, y: point.y - 0.04, class: 'landmark-label'});
            label.textContent = point.label;
            svg.appendChild(label);
        }
    });

    const byName = new Map([...data.sources, ...data.detectors].map(point => [point.name, point]));
    data.pairs.forEach(pair => {
        const source = byName.get(pair.source);
        const detector = byName.get(pair.detector);
        if (!source || !detector) return;
        if (data.showLines) {
            svg.appendChild(make('line', {x1: source.x, y1: source.y, x2: detector.x, y2: detector.y, class: 'channel-line'}));
        }
        if (data.showMidpoints) {
            svg.appendChild(make('circle', {
                cx: (source.x + detector.x) / 2,
                cy: (source.y + detector.y) / 2,
                r: 0.022,
                class: 'midpoint'
            }));
        }
    });

    data.points.forEach(point => {
        const group = make('g', {class: 'slot'});
        group.appendChild(make('circle', {cx: point.x, cy: point.y, r: 0.014, class: 'slot-dot'}));
        const hit = make('circle', {cx: point.x, cy: point.y, r: 0.021, class: 'slot-hit'});
        const title = make('title');
        title.textContent = point.label;
        hit.appendChild(title);
        hit.onclick = event => {
            event.stopPropagation();
            setTriggerValue('clicked', {label: point.label, nonce: Date.now()});
        };
        group.appendChild(hit);
        svg.appendChild(group);
    });

    const drawOptodes = (points, kind, visible) => {
        if (!visible) return;
        points.forEach(point => {
            svg.appendChild(make('circle', {cx: point.x, cy: point.y, r: 0.026, class: kind}));
            const label = make('text', {x: point.x, y: point.y - 0.04, class: 'optode-label'});
            label.textContent = point.name;
            svg.appendChild(label);
        });
    };
    drawOptodes(data.sources, 'source', data.showSources);
    drawOptodes(data.detectors, 'detector', data.showDetectors);
}
"""


@lru_cache(maxsize=1)
def _get_head_map_component() -> object:
    import streamlit as st

    return st.components.v2.component(
        "montage_head_map",
        html=_HEAD_MAP_HTML,
        css=_HEAD_MAP_CSS,
        js=_HEAD_MAP_JS,
    )


def app() -> None:
    try:
        import streamlit as st
    except ImportError as error:
        raise SystemExit('Install montage-align with: pip install -e .') from error

    st.set_page_config(page_title="MtSi1005 - Montage Alignment Toolkit", layout="wide")
    st.title("MtSi1005 - Montage Alignment Toolkit")
    st.caption("Import geometry, adjust its mapping, review it in 3D, and export locally.")
    stages = ["1. Import", "2. Adjust & Map", "3. Review & Export"]
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] button,
        [data-testid="stAppViewContainer"] input,
        [data-testid="stAppViewContainer"] textarea {
            font-size: 14pt;
        }
        [data-testid="stAppViewContainer"] h1 { font-size: 44px !important; }
        [data-testid="stAppViewContainer"] h2 { font-size: 36px !important; }
        [data-testid="stAppViewContainer"] h3 { font-size: 28px !important; }
        [data-testid="stAppViewContainer"] h4 { font-size: 24px !important; }
        [data-testid="stAppViewContainer"] h5 { font-size: 20px !important; }
        [data-testid="stAppViewContainer"] h6 { font-size: 16px !important; }
        [data-testid="stAppViewContainer"] button *,
        [data-baseweb="tag"] *,
        [data-baseweb="menu"] *,
        [data-baseweb="select"] *,
        [role="radio"] * {
            font-size: 14pt !important;
        }
        [data-testid="stDataFrameGlideDataEditor"] {
            --gdg-editor-font-size: 14pt !important;
        }
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            font-size: 14pt !important;
        }
        [class*="st-key-workflow_stage"] [role="radiogroup"] {
            width: 100%;
        }
        [class*="st-key-workflow_stage"] button[role="radio"] {
            flex: 1 1 0;
            justify-content: center;
            background: #f3f4f6 !important;
            border-color: #d1d5db !important;
            color: #374151 !important;
        }
        [class*="st-key-workflow_stage"] button[role="radio"][aria-checked="true"] {
            background: #0f766e !important;
            border-color: #0f766e !important;
            color: #ffffff !important;
        }
        [class*="st-key-workflow_stage"] button[role="radio"]:hover {
            background: #e5e7eb !important;
            border-color: #9ca3af !important;
        }
        [class*="st-key-workflow_stage"] button[role="radio"][aria-checked="true"]:hover {
            background: #115e59 !important;
            border-color: #115e59 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = stages[0]
    stage = st.segmented_control(
        "Workflow stage",
        stages,
        key="workflow_stage",
        selection_mode="single",
        width="stretch",
    )
    if stage == stages[0]:
        _import_geometry(st)
        _workflow_footer(st, previous=None, next_stage=stages[1])
    elif stage == stages[1]:
        _adjust_and_map(st)
        _workflow_footer(st, previous=stages[0], next_stage=stages[2])
    else:
        _review_and_export(st)
        _workflow_footer(st, previous=stages[1], next_stage=None)
    st.markdown(
        '<p style="margin-top:2.5rem;text-align:center;color:#9ca3af;font-size:0.8rem;font-style:italic">'
        'Built by Tony Chae during a 2026 Microsoft Research internship.'
        '</p>',
        unsafe_allow_html=True,
    )


def _set_workflow_stage(state, stage: str) -> None:
    state.workflow_stage = stage


def _workflow_footer(st, *, previous: str | None, next_stage: str | None) -> None:
    st.divider()
    st.markdown(
        """
        <style>
        @keyframes workflow-next-ready {
            0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
            50% { box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.18); }
        }
        [class*="st-key-workflow_next_"] button {
            background: #ecfdf5 !important;
            border-color: #bbf7d0 !important;
            color: #6b7280 !important;
        }
        [class*="st-key-workflow_next_"] button:not(:disabled) {
            background: #dcfce7 !important;
            border-color: #86efac !important;
            color: #14532d !important;
            animation: workflow-next-ready 2.4s ease-in-out infinite;
        }
        [class*="st-key-workflow_next_"] button:not(:disabled):hover {
            background: #bbf7d0 !important;
            border-color: #4ade80 !important;
        }
        @media (prefers-reduced-motion: reduce) {
            [class*="st-key-workflow_next_"] button { animation: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    previous_column, spacer, next_column = st.columns([1, 2, 1])
    if previous is not None:
        previous_column.button(
            "← Previous step",
            on_click=_set_workflow_stage,
            args=(st.session_state, previous),
            width="stretch",
        )
    if next_stage is not None:
        needs_import = next_stage == "2. Adjust & Map" and not st.session_state.get("imported_batch")
        needs_result = next_stage == "3. Review & Export" and not (
            st.session_state.get("alignment_results") or st.session_state.get("alignment_result")
        )
        next_column.button(
            f"Next step: {next_stage.split('. ', 1)[1]}",
            key=f"workflow_next_{next_stage.split('. ', 1)[0]}",
            type="primary",
            disabled=needs_import or needs_result,
            on_click=_set_workflow_stage,
            args=(st.session_state, next_stage),
            width="stretch",
        )


def _example_input_root() -> Path:
    source_root = Path(__file__).resolve().parents[3] / "example_test_input"
    return source_root if source_root.is_dir() else Path(__file__).resolve().parent / "example_test_input"


def _remap_sudan2026_axes(montage: MontageInput) -> MontageInput:
    def remap(positions: np.ndarray) -> np.ndarray:
        return np.column_stack((-positions[:, 0], positions[:, 2], -positions[:, 1]))

    optodes = montage.optodes
    if optodes is not None:
        optodes = OptodeGeometry(
            optodes.source_names,
            remap(optodes.source_positions),
            optodes.detector_names,
            remap(optodes.detector_positions),
            optodes.channel_pairs,
        )
    return MontageInput(
        montage.names,
        remap(montage.positions),
        montage.unit,
        "head",
        montage.modality,
        optodes,
    )


def _load_geometry_path(path: Path, *, unit: str, frame: str) -> MontageInput:
    suffix = path.suffix.lower()
    if suffix == ".snirf":
        montage = load_snirf(
            path,
            physical_pairs=True,
            min_sd_distance_mm=_DEFAULT_SD_MIN_MM,
            max_sd_distance_mm=_DEFAULT_SD_MAX_MM,
        )
        if "sudan2026" in str(path).lower():
            return _remap_sudan2026_axes(montage)
        return montage
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".tsv":
        return load_positions_tsv(path, unit=unit, frame=frame)
    if suffix == ".txt":
        return load_position_text(path, unit=unit, frame=frame)
    if suffix == ".xdf":
        return load_xdf(path)
    raise ValueError("Supported inputs are TSV, TXT, JSON, NPZ, SNIRF, and XDF")


def _canonical_labels_from_path(path: Path) -> tuple[str, ...] | None:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "canonical_labels" not in data:
                return None
            return tuple(np.asarray(data["canonical_labels"]).astype(str))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload.get("canonical_labels")
        return tuple(map(str, labels)) if labels is not None else None
    return None


def _set_imported_batch(st, entries: list[dict], report: list[dict[str, str]] | None = None) -> None:
    st.session_state.imported_batch = entries
    st.session_state.import_scan_report = report or []
    st.session_state.alignment_results = {}
    st.session_state.pop("alignment_result", None)


def _missing_coordinate_inputs(entries: list[dict]) -> dict[str, tuple[str, ...]]:
    missing: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        if entry.get("canonical_labels") is not None:
            continue
        montage = entry["montage"]
        invalid = ~np.isfinite(montage.positions).all(axis=1)
        if invalid.any():
            missing[entry["label"]] = tuple(np.asarray(montage.names)[invalid])
    return missing


def _parsed_geometry_preview_rows(
    montage: MontageInput,
    canonical_labels: tuple[str, ...] | None,
) -> list[dict[str, object]]:
    axes = ("x", "y", "z") if montage.positions.shape[1] == 3 else ("x", "y")
    rows: list[dict[str, object]] = []
    for index, name in enumerate(montage.names):
        row: dict[str, object] = {"name": name}
        if canonical_labels is not None:
            row["canonical_label"] = canonical_labels[index]
        row.update(
            {
                axis: float(montage.positions[index, axis_index])
                for axis_index, axis in enumerate(axes)
            }
        )
        rows.append(row)
    return rows


def _style_placeholder_coordinates(rows: list[dict[str, object]], dimensions: int):
    import pandas as pd

    axes = list(("x", "y", "z")[:dimensions])
    return pd.DataFrame(rows).style.set_properties(
        subset=axes,
        color="#a6adb8",
    )


_BIDS_SIDECAR_SUFFIXES = ("_optodes.tsv", "_coordsystem.json", "_channels.tsv")
_SCANNED_INPUT_SUFFIXES = {".tsv", ".txt", ".json", ".npz", ".snirf", ".xdf", ".mat", ".nirs"}


def _bids_entities(path: Path, suffix: str) -> dict[str, str]:
    prefix = path.name[: -len(suffix)]
    entities: dict[str, str] = {}
    for token in prefix.split("_"):
        if "-" in token:
            key, value = token.split("-", 1)
            entities[key] = value
    return entities


def _compatible_bids_sidecar(channels_path: Path, candidates: list[Path], suffix: str) -> Path | None:
    channel_entities = _bids_entities(channels_path, "_channels.tsv")
    compatible = []
    for candidate in candidates:
        candidate_entities = _bids_entities(candidate, suffix)
        if all(channel_entities.get(key) == value for key, value in candidate_entities.items()):
            common_depth = len(Path(*Path(channels_path.parent).parts).parents)
            try:
                common_depth = len(Path(*Path(channels_path.parent).relative_to(candidate.parent).parts).parts)
                ancestor_score = -common_depth
            except ValueError:
                ancestor_score = -1000
            compatible.append((len(candidate_entities), ancestor_score, candidate))
    return max(compatible, default=(0, 0, None), key=lambda item: item[:2])[2]


def _discover_input_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink()) and path.suffix.lower() in _SCANNED_INPUT_SUFFIXES
    )


def _table_columns(path: Path) -> set[str]:
    try:
        first_line = path.open(encoding="utf-8-sig", errors="replace").readline().strip()
    except OSError:
        return set()
    return {column.strip().lower() for column in re.split(r"[\t,;]", first_line)}


def _standalone_kind(path: Path, modality: str) -> tuple[str, str]:
    suffix = path.suffix.lower()
    lowered = path.name.lower()
    parent_parts = {part.lower() for part in path.parts}
    if modality == "fNIRS" and suffix != ".xdf" and "eeg" in parent_parts:
        return "skipped", "EEG file excluded by modality filter"
    if modality == "EEG" and suffix != ".xdf" and "nirs" in parent_parts:
        return "skipped", "fNIRS file excluded by modality filter"
    if lowered.endswith(_BIDS_SIDECAR_SUFFIXES):
        return "sidecar", "BIDS sidecar"
    if suffix in {".snirf", ".xdf"}:
        return "candidate", suffix.removeprefix(".").upper()
    if suffix in {".mat", ".nirs"}:
        return "warning", f"{suffix} needs a dataset/device adapter"
    if suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                valid = "ch_names" in data and ("ch_pos" in data or "ch_pos_1005" in data)
        except Exception:
            valid = False
        return ("candidate", "geometry NPZ") if valid else ("skipped", "NPZ has no geometry schema")
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            valid = isinstance(payload, dict) and "ch_names" in payload and (
                "ch_pos" in payload or "ch_pos_1005" in payload
            )
        except (OSError, ValueError):
            valid = False
        return ("candidate", "geometry JSON") if valid else ("skipped", "JSON is metadata, not geometry")
    if suffix == ".tsv":
        columns = _table_columns(path)
        name_columns = {"name", "channel", "ch_name", "label", "optode"}
        if columns & name_columns and {"x", "y"}.issubset(columns):
            return "candidate", "position table"
        return "skipped", "TSV has no position columns"
    if suffix == ".txt":
        try:
            load_position_text(path)
        except (OSError, ValueError):
            return "skipped", "TXT is not a coordinate table"
        return "candidate", "position text"
    return "skipped", "unsupported file"


def _is_nirs_channels(path: Path) -> bool:
    columns = _table_columns(path)
    if {"source", "detector"}.issubset(columns):
        return True
    try:
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))[:20]
    except OSError:
        return False
    return any(str(row.get("type", "")).upper().startswith("NIRS") for row in rows)


def _montage_signature(montage: MontageInput) -> tuple:
    return (
        montage.names,
        montage.positions.shape,
        montage.positions.astype(np.float64).round(9).tobytes(),
        montage.unit,
        montage.frame,
    )


def _scan_path_batch(
    input_paths: list[Path], *, unit: str, frame: str, modality: str = "fNIRS"
) -> tuple[list[dict], list[dict[str, str]]]:
    entries: list[dict] = []
    report: list[dict[str, str]] = []
    paths = list(dict.fromkeys(Path(path) for path in input_paths))
    optodes_paths = [path for path in paths if path.name.lower().endswith("_optodes.tsv")]
    coordsystem_paths = [path for path in paths if path.name.lower().endswith("_coordsystem.json")]
    all_channels_paths = [path for path in paths if path.name.lower().endswith("_channels.tsv")]
    derivative_channels = [
        path for path in all_channels_paths if "derivatives" in {part.lower() for part in path.parts}
    ]
    channels_paths = (
        [
            path
            for path in all_channels_paths
            if path not in derivative_channels and _is_nirs_channels(path)
        ]
        if modality != "EEG"
        else []
    )
    for path in derivative_channels:
        report.append({"status": "Skipped", "file": str(path), "reason": "derivative/QC table"})
    for path in set(all_channels_paths) - set(channels_paths):
        report.append({"status": "Skipped", "file": str(path), "reason": "non-NIRS channels table"})
    consumed: set[Path] = set()
    probe_groups: dict[Path, list[Path]] = {}
    for path in paths:
        if path.suffix.lower() == ".txt" and re.match(r"(?i)^(?:PS|probe)[-_]?\d+\.txt$", path.name):
            probe_groups.setdefault(path.parent, []).append(path)
    for parent, group in probe_groups.items():
        if len(group) < 2:
            continue
        try:
            montages = [(path, load_position_text(path, unit=unit, frame=frame)) for path in sorted(group)]
            names = tuple(
                f"{path.stem}_{name}"
                for path, montage in montages
                for name in montage.names
            )
            positions = np.vstack([montage.positions for _, montage in montages])
            label_root = parent.parent.name if parent.name.lower() == "raw" else parent.name
            entries.append(
                {
                    "label": f"{label_root}-probe-set",
                    "montage": MontageInput(names, positions, unit, frame),
                    "canonical_labels": None,
                }
            )
            consumed.update(group)
            report.append(
                {
                    "status": "Accepted",
                    "file": str(parent),
                    "reason": f"grouped {len(group)} numbered probe coordinate files",
                }
            )
        except Exception as error:
            report.append({"status": "Warning", "file": str(parent), "reason": str(error)})
    for channels in channels_paths:
        optodes_path = _compatible_bids_sidecar(channels, optodes_paths, "_optodes.tsv")
        coordsystem = _compatible_bids_sidecar(channels, coordsystem_paths, "_coordsystem.json")
        label = channels.name[: -len("_channels.tsv")]
        if optodes_path is None or coordsystem is None:
            same_directory_recording = next(
                (path for path in paths if path.parent == channels.parent and path.suffix.lower() == ".snirf"),
                None,
            )
            if same_directory_recording is not None:
                report.append(
                    {
                        "status": "Skipped",
                        "file": str(channels),
                        "reason": f"channel metadata represented by {same_directory_recording.name}",
                    }
                )
            else:
                report.append(
                    {
                        "status": "Warning",
                        "file": str(channels),
                        "reason": "NIRS channels table has no geometry source (optodes sidecar or SNIRF)",
                    }
                )
            consumed.add(channels)
            continue
        try:
            montage = load_bids_nirs_sidecars(
                optodes_path,
                coordsystem,
                channels,
                min_sd_distance_mm=_DEFAULT_SD_MIN_MM,
                max_sd_distance_mm=_DEFAULT_SD_MAX_MM,
            )
            source_files = [str(channels), str(optodes_path), str(coordsystem)]
            channel_entities = _bids_entities(channels, "_channels.tsv")
            for candidate in paths:
                if candidate.suffix.lower() != ".snirf":
                    continue
                snirf_entities = _bids_entities(candidate, ".snirf")
                if channel_entities and all(snirf_entities.get(key) == value for key, value in channel_entities.items()):
                    consumed.add(candidate)
                    source_files.append(str(candidate))
            entries.append(
                {
                    "label": label,
                    "montage": montage,
                    "canonical_labels": None,
                    "source_files": source_files,
                }
            )
            report.append({"status": "Accepted", "file": str(channels), "reason": "BIDS NIRS montage"})
            consumed.update({optodes_path, coordsystem, channels})
        except Exception as error:
            report.append({"status": "Warning", "file": str(channels), "reason": str(error)})

    for path in paths:
        if path in consumed:
            continue
        lowered = path.name.lower()
        if lowered.endswith(_BIDS_SIDECAR_SUFFIXES):
            if path not in consumed:
                report.append({"status": "Skipped", "file": str(path), "reason": "unused BIDS sidecar"})
            continue
        kind, reason = _standalone_kind(path, modality)
        if kind == "skipped":
            report.append({"status": "Skipped", "file": str(path), "reason": reason})
            continue
        if kind == "warning":
            report.append({"status": "Warning", "file": str(path), "reason": reason})
            continue
        try:
            montage = _load_geometry_path(path, unit=unit, frame=frame)
            entries.append(
                {
                    "label": path.stem,
                    "montage": montage,
                    "canonical_labels": _canonical_labels_from_path(path),
                    "source_files": [str(path)],
                }
            )
            report.append({"status": "Accepted", "file": str(path), "reason": reason})
        except Exception as error:
            report.append({"status": "Warning", "file": str(path), "reason": str(error)})

    unique_entries: list[dict] = []
    seen_signatures: dict[tuple, dict] = {}
    for entry in entries:
        entry.setdefault("source_files", [entry["label"]])
        signature = _montage_signature(entry["montage"])
        if signature in seen_signatures:
            representative = seen_signatures[signature]
            representative["source_files"].extend(entry["source_files"])
            report.append(
                {
                    "status": "Skipped",
                    "file": entry["label"],
                    "reason": f"duplicate montage of {representative['label']}",
                }
            )
            continue
        seen_signatures[signature] = entry
        unique_entries.append(entry)
    entries = unique_entries
    if entries:
        for row in report:
            if row["status"] == "Warning" and "needs a dataset/device adapter" in row["reason"]:
                row["status"] = "Skipped"
                row["reason"] = "alternative vendor file; recognized geometry already found"
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)):
        for index, entry in enumerate(entries):
            entry["label"] = f"{entry['label']}-{index + 1}"
    return entries, report


def _load_path_batch(
    input_paths: list[Path], *, unit: str, frame: str, modality: str = "fNIRS"
) -> list[dict]:
    entries, _ = _scan_path_batch(input_paths, unit=unit, frame=frame, modality=modality)
    return entries


def _scan_uploaded_batch(
    uploaded_files, *, unit: str, frame: str, modality: str = "fNIRS"
) -> tuple[list[dict], list[dict[str, str]]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for uploaded in uploaded_files:
            if Path(uploaded.name).suffix.lower() == ".zip":
                with zipfile.ZipFile(io.BytesIO(uploaded.getvalue())) as archive:
                    for member in archive.infolist():
                        relative = Path(member.filename)
                        if member.is_dir() or relative.is_absolute() or ".." in relative.parts:
                            continue
                        target = root / Path(uploaded.name).stem / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(member))
            else:
                relative = Path(uploaded.name)
                if relative.is_absolute() or ".." in relative.parts:
                    relative = Path(relative.name)
                target = root / "uploads" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded.getvalue())
        return _scan_path_batch(
            _discover_input_paths(root), unit=unit, frame=frame, modality=modality
        )


def _load_uploaded_batch(
    uploaded_files, *, unit: str, frame: str, modality: str = "fNIRS"
) -> list[dict]:
    entries, _ = _scan_uploaded_batch(
        uploaded_files, unit=unit, frame=frame, modality=modality
    )
    return entries


def _scan_report_summary(st, report: list[dict[str, str]]) -> None:
    if not report:
        return
    counts = {
        status: sum(row["status"] == status for row in report)
        for status in ("Accepted", "Skipped", "Warning")
    }
    columns = st.columns(3)
    columns[0].metric("Accepted files", counts["Accepted"])
    columns[1].metric("Auto-skipped", counts["Skipped"])
    columns[2].metric("Warnings", counts["Warning"])
    warnings = [row for row in report if row["status"] == "Warning"]
    if warnings:
        st.warning(f"{len(warnings)} montage-like files need attention; they were not aligned.")
        warning_groups: dict[str, dict[str, object]] = {}
        for row in warnings:
            group = warning_groups.setdefault(
                row["reason"], {"reason": row["reason"], "files": 0, "example": row["file"]}
            )
            group["files"] = int(group["files"]) + 1
        st.dataframe(list(warning_groups.values()), hide_index=True, width="stretch")
    skipped_reasons: dict[str, int] = {}
    for row in report:
        if row["status"] == "Skipped":
            skipped_reasons[row["reason"]] = skipped_reasons.get(row["reason"], 0) + 1
    with st.expander("Skipped-file summary"):
        st.dataframe(
            [{"reason": reason, "files": count} for reason, count in sorted(skipped_reasons.items())],
            hide_index=True,
            width="stretch",
        )
    with st.expander("All scanned files and decisions"):
        st.dataframe(report, hide_index=True, width="stretch")


def _import_geometry(st) -> None:
    st.subheader("Geometry input")
    source = st.segmented_control(
        "Source",
        ["Upload files", "Upload folder", "Example test input"],
        default="Upload files",
        selection_mode="single",
    )
    unit_column, modality_column = st.columns(2)
    with unit_column:
        unit = st.selectbox("Unit for TSV/TXT", ["m", "cm", "mm"], index=0)
    with modality_column:
        modality = st.selectbox("Montage modality", ["fNIRS", "EEG", "Auto"])
    frame = "head"
    st.caption(
        "Non-montage files are auto-skipped, never modified or deleted. "
        "Temporary upload extraction is removed automatically after scanning."
    )
    if modality != "EEG":
        st.caption("fNIRS SNIRF/BIDS physical pairs are filtered to 8–35 mm during import.")

    if source in {"Upload files", "Upload folder"}:
        directory_mode = source == "Upload folder"
        uploaded = st.file_uploader(
            "BIDS or geometry folder" if directory_mode else "Geometry files, BIDS sidecars, or a ZIP folder",
            type=["tsv", "txt", "json", "npz", "snirf", "xdf", "zip"],
            accept_multiple_files="directory" if directory_mode else True,
            key="folder-upload" if directory_mode else "file-upload",
        )
        if directory_mode:
            st.caption("Choose the BIDS root once. Its nested relative paths are preserved and scanned recursively.")
        else:
            st.caption("A single TSV works directly. ZIP preserves nested BIDS folders for batch upload.")
        button_label = "Load selected folder" if directory_mode else "Load file batch"
        if st.button(button_label, type="primary", disabled=not uploaded):
            try:
                entries, report = _scan_uploaded_batch(
                    uploaded, unit=unit, frame=frame, modality=modality
                )
                _set_imported_batch(st, entries, report)
            except Exception as error:
                st.error(str(error))
    else:
        root = _example_input_root()
        examples = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        if not examples:
            st.warning("Synthetic example test inputs are not installed in this environment.")
        else:
            selected = st.multiselect(
                "Synthetic example scenarios",
                examples,
                default=examples,
                format_func=lambda path: path.stem,
                key="example-datasets-" + str(hash(tuple(path.name for path in examples))),
            )
            st.caption(
                f"{len(selected)} package-authored synthetic scenarios selected. "
                "No patient or pretraining-dataset geometry is included."
            )
            if st.button("Load example test input", type="primary", disabled=not selected):
                try:
                    entries: list[dict] = []
                    report: list[dict[str, str]] = []
                    for directory in selected:
                        directory_entries, directory_report = _scan_path_batch(
                            _discover_input_paths(directory),
                            unit=unit,
                            frame=frame,
                            modality=modality,
                        )
                        entries.extend(directory_entries)
                        report.extend(directory_report)
                    _set_imported_batch(st, entries, report)
                except Exception as error:
                    st.error(str(error))

    batch = st.session_state.get("imported_batch", [])
    report = st.session_state.get("import_scan_report", [])
    if report and not batch:
        st.error("No recognized montage geometry was found.")
        _scan_report_summary(st, report)
    if batch:
        st.success(f"Loaded {len(batch)} montages with {sum(len(entry['montage'].names) for entry in batch)} total inputs")
        _scan_report_summary(st, report)
        missing_inputs = _missing_coordinate_inputs(batch)
        if missing_inputs:
            details = "; ".join(
                f"{label}: {', '.join(names)}" for label, names in missing_inputs.items()
            )
            st.error(
                "Missing or non-finite coordinates detected. These inputs cannot be aligned "
                f"until corrected in Adjust & Map. {details}"
            )
        large_montages = [
            f"{entry['label']} ({len(entry['montage'].names)} pairs)"
            for entry in batch
            if len(entry["montage"].names) > 512
        ]
        if large_montages:
            st.info(
                "Large physical-pair montages detected: "
                + ", ".join(large_montages)
                + ". This can be valid for HD-DOT; review the pair set before alignment."
            )
        st.dataframe(
            [
                {
                    "montage": entry["label"],
                    "files": len(entry.get("source_files", [entry["label"]])),
                    "inputs": len(entry["montage"].names),
                    "dimensions": entry["montage"].positions.shape[1],
                    "unit": entry["montage"].unit,
                    "frame": entry["montage"].frame,
                    "canonical_labels": entry["canonical_labels"] is not None,
                }
                for entry in batch
            ],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Files represented by each montage"):
            for entry in batch:
                st.markdown(f"**{entry['label']}**")
                st.code("\n".join(entry.get("source_files", [entry["label"]])))
        preview_label = st.selectbox(
            "Preview parsed montage",
            [entry["label"] for entry in batch],
            key="import-montage-preview",
        )
        preview_entry = next(entry for entry in batch if entry["label"] == preview_label)
        preview_montage = preview_entry["montage"]
        preview_rows = _parsed_geometry_preview_rows(
            preview_montage,
            preview_entry["canonical_labels"],
        )
        st.caption(
            "Parsed geometry preview only; SNIRF/XDF signal samples are not displayed or retained."
        )
        preview_table = (
            _style_placeholder_coordinates(preview_rows, preview_montage.positions.shape[1])
            if preview_entry["canonical_labels"] is not None
            else preview_rows
        )
        st.dataframe(preview_table, hide_index=True, width="stretch")
        if preview_montage.optodes is not None:
            st.caption(
                f"Optodes preserved: {len(preview_montage.optodes.source_names)} sources, "
                f"{len(preview_montage.optodes.detector_names)} detectors."
            )


def _apply_axis_selections(positions: np.ndarray, selections: list[str]) -> np.ndarray:
    axis_index = {"x": 0, "y": 1, "z": 2}
    return np.column_stack(
        [
            positions[:, axis_index[selection[-1]]] * (-1 if selection[0] == "-" else 1)
            for selection in selections
        ]
    )


def _axis_transform(st, positions: np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    if positions.shape[1] != 3 or not st.toggle("Apply axis permutation or flip", value=False):
        return positions, None
    options = ["+x", "-x", "+y", "-y", "+z", "-z"]
    columns = st.columns(3)
    selections = []
    for column, target, default in zip(columns, ("right x", "front y", "up z"), (0, 2, 4)):
        with column:
            selections.append(st.selectbox(target, options, index=default))
    source_axes = [selection[-1] for selection in selections]
    if len(set(source_axes)) != 3:
        raise ValueError("Each source axis must be used exactly once")
    return _apply_axis_selections(positions, selections), selections


def _project_positions_2d(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[1] == 2:
        return positions.copy()
    output = np.full((len(positions), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(positions).all(axis=1)
    points = positions[valid]
    radii = np.linalg.norm(points, axis=1, keepdims=True)
    radii[radii == 0] = 1.0
    unit = points / radii
    polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    output[valid, 0] = polar * np.cos(azimuth)
    output[valid, 1] = polar * np.sin(azimuth)
    return output


def _alignment_2d_figure(
    result: AlignmentResult,
    *,
    montage: MontageInput | None = None,
    show_reference: bool = True,
    show_midpoints: bool = True,
    show_assigned_slots: bool = True,
    show_assignment_lines: bool = True,
    show_sources: bool = False,
    show_detectors: bool = False,
    show_sd_lines: bool = False,
):
    import plotly.graph_objects as go

    reference = load_reference(result.reference)
    reference_labels = np.asarray(reference["labels"])
    reference_2d = np.asarray(reference["positions_2d"])
    valid = result.valid
    midpoint_2d = _project_positions_2d(result.source_positions)[valid]
    assigned_2d = reference_2d[result.slot_ids[valid]]
    radius = float(np.nanmax(np.linalg.norm(reference_2d, axis=1))) * 1.04
    theta = np.linspace(0, 2 * np.pi, 180)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            mode="lines",
            line={"color": "#7f8b86", "width": 2},
            hoverinfo="skip",
            name="Head outline",
            showlegend=False,
        )
    )
    if show_reference:
        figure.add_trace(
            go.Scatter(
                x=reference_2d[:, 0],
                y=reference_2d[:, 1],
                mode="markers",
                marker={"size": 5, "color": "#aeb8b3", "opacity": 0.65},
                text=reference_labels,
                hovertemplate="%{text}<extra>Standard 1005</extra>",
                name="Standard 1005",
            )
        )
    nose_width = radius * 0.10
    nose_height = radius * 0.13
    figure.add_trace(
        go.Scatter(
            x=[-nose_width, 0.0, nose_width],
            y=[radius * 0.98, radius + nose_height, radius * 0.98],
            mode="lines",
            line={"color": "#7f8b86", "width": 2},
            hovertemplate="ANTERIOR (+y)<extra>Nose</extra>",
            name="Nose (front)",
            showlegend=False,
        )
    )
    ear_theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
    for side, name in ((-1.0, "Left ear"), (1.0, "Right ear")):
        figure.add_trace(
            go.Scatter(
                x=side * (radius + radius * 0.055 * np.cos(ear_theta)),
                y=radius * 0.18 * np.sin(ear_theta),
                mode="lines",
                line={"color": "#7f8b86", "width": 2},
                hoverinfo="skip",
                name=name,
                showlegend=False,
            )
        )
    if montage is not None and montage.optodes is not None:
        optodes = montage.optodes
        sources = _project_positions_2d(
            transform_positions(montage, optodes.source_positions, reference=result.reference)
        )
        detectors = _project_positions_2d(
            transform_positions(montage, optodes.detector_positions, reference=result.reference)
        )
        if show_sd_lines:
            line_x, line_y = [], []
            for source_index, detector_index in optodes.channel_pairs:
                line_x.extend([sources[source_index, 0], detectors[detector_index, 0], None])
                line_y.extend([sources[source_index, 1], detectors[detector_index, 1], None])
            figure.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line={"color": "#65716d", "width": 1},
                    opacity=0.55,
                    hoverinfo="skip",
                    name="S-D lines",
                )
            )
        for visible, positions, names, color, trace_name in (
            (show_sources, sources, optodes.source_names, "#d62728", "Sources"),
            (show_detectors, detectors, optodes.detector_names, "#1f77b4", "Detectors"),
        ):
            if visible:
                figure.add_trace(
                    go.Scatter(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        mode="markers",
                        marker={"size": 9, "color": color, "line": {"color": "white", "width": 1}},
                        text=names,
                        hovertemplate="%{text}<extra>" + trace_name + "</extra>",
                        name=trace_name,
                    )
                )
    if show_assignment_lines:
        line_x, line_y = [], []
        for midpoint, assigned in zip(midpoint_2d, assigned_2d):
            line_x.extend([midpoint[0], assigned[0], None])
            line_y.extend([midpoint[1], assigned[1], None])
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "#9aa49f", "width": 1},
                hoverinfo="skip",
                name="Midpoint to slot",
            )
        )
    if show_midpoints:
        figure.add_trace(
            go.Scatter(
                x=midpoint_2d[:, 0],
                y=midpoint_2d[:, 1],
                mode="markers",
                marker={"size": 6, "color": "#f2b51d", "line": {"color": "#493d23", "width": 1}},
                text=np.asarray(result.names)[valid],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra>Aligned midpoint</extra>",
                name="Aligned midpoints",
            )
        )
    if show_assigned_slots:
        figure.add_trace(
            go.Scatter(
                x=assigned_2d[:, 0],
                y=assigned_2d[:, 1],
                mode="markers",
                marker={
                    "size": 9,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#111111", "width": 1.5},
                },
                text=np.asarray(result.slot_labels)[valid],
                hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra>Assigned slot</extra>",
                name="Assigned slots",
            )
        )
    figure.update_layout(
        height=560,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        dragmode=False,
        xaxis={
            "title": "right (+x), projection",
            "scaleanchor": "y",
            "scaleratio": 1,
            "fixedrange": True,
            "range": [-radius * 1.18, radius * 1.18],
        },
        yaxis={
            "title": "front (+y), projection",
            "fixedrange": True,
            "range": [-radius * 1.18, radius * 1.18],
        },
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
    )
    return figure


def _alignment_2d_rows(result: AlignmentResult) -> list[dict[str, object]]:
    reference_2d = np.asarray(load_reference(result.reference)["positions_2d"])
    source_2d = _project_positions_2d(result.source_positions)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(result.names):
        slot_id = int(result.slot_ids[index])
        slot_position = reference_2d[slot_id] if slot_id >= 0 else np.asarray([np.nan, np.nan])
        rows.append(
            {
                "channel": name,
                "midpoint_x": float(source_2d[index, 0]),
                "midpoint_y": float(source_2d[index, 1]),
                "assigned_slot": result.slot_labels[index],
                "assigned_slot_x": float(slot_position[0]),
                "assigned_slot_y": float(slot_position[1]),
                "distance": float(result.distances[index]),
                "valid": bool(result.valid[index]),
            }
        )
    return rows


def _adjust_imported_geometry(st) -> None:
    batch = st.session_state.get("imported_batch", [])
    if not batch:
        st.info("Import one or more geometry files in the first tab.")
        return

    action_column, status_column = st.columns([1, 2])
    with action_column:
        if st.button("Align all automatically", type="primary"):
            try:
                results = {
                    entry["label"]: align(
                        entry["montage"], canonical_labels=entry["canonical_labels"]
                    )
                    for entry in batch
                }
                st.session_state.alignment_results = results
                st.session_state.alignment_montages = {
                    entry["label"]: entry["montage"] for entry in batch
                }
                st.session_state.alignment_result = next(iter(results.values()))
                st.success(f"Aligned {len(results)} montages")
            except Exception as error:
                st.error(str(error))
    with status_column:
        st.caption("Automatic batch mode uses each file's declared unit/frame and canonical labels. Select one montage below for corrections.")

    labels = [entry["label"] for entry in batch]
    selected_label = st.selectbox("Montage to inspect or adjust", labels)
    entry = next(entry for entry in batch if entry["label"] == selected_label)
    montage = entry["montage"]
    st.session_state.selected_montage_label = selected_label
    represented_files = entry.get("source_files", [entry["label"]])
    st.caption(
        f"This montage represents {len(represented_files)} input file"
        + ("." if len(represented_files) == 1 else "s with identical geometry.")
    )
    with st.expander("Represented patient / recording files"):
        st.code("\n".join(represented_files))
    st.subheader(selected_label)
    imported_labels = entry["canonical_labels"]
    dimensions = montage.positions.shape[1]
    editor_positions = montage.positions.copy()
    use_labels = st.toggle(
        "Override geometry with canonical labels",
        value=imported_labels is not None,
    )
    rows = []
    for index, name in enumerate(montage.names):
        row = {"name": name, "x": editor_positions[index, 0], "y": editor_positions[index, 1]}
        if dimensions == 3:
            row["z"] = editor_positions[index, 2]
        rows.append(row)
    if use_labels:
        st.dataframe(
            _style_placeholder_coordinates(rows, dimensions),
            hide_index=True,
            width="stretch",
        )
        st.caption("Coordinates show the original input values and are ignored while canonical override is active.")
        records = rows
    else:
        edited = st.data_editor(
            rows,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            key=f"geometry-editor-{selected_label}",
        )
        records = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
    names = tuple(str(row["name"]).strip() for row in records)
    axes = ("x", "y", "z") if dimensions == 3 else ("x", "y")
    positions = np.asarray([[row[axis] for axis in axes] for row in records], dtype=np.float64)
    missing_names = tuple(name for name, position in zip(names, positions) if not np.isfinite(position).all())
    if not use_labels and missing_names:
        st.error(
            "Missing or non-finite coordinates must be corrected before alignment: "
            + ", ".join(missing_names)
        )

    setting_columns = st.columns(2)
    with setting_columns[0]:
        unit = st.selectbox("Coordinate unit", ["m", "cm", "mm"], index=["m", "cm", "mm"].index(montage.unit))
    with setting_columns[1]:
        reference = st.selectbox("Target reference", ["standard_1005", "standard_1020"])
    frame = montage.frame
    st.caption(f"Input coordinate frame: {frame} (read from the imported geometry)")

    try:
        positions, axis_selections = _axis_transform(st, positions)
    except ValueError as error:
        st.error(str(error))
        return

    canonical_labels = None
    if use_labels:
        reference_labels = list(load_reference(reference)["labels"])
        defaults = imported_labels or tuple("" for _ in names)
        label_rows = [
            {"name": name, "canonical_label": defaults[index] if index < len(defaults) else ""}
            for index, name in enumerate(names)
        ]
        edited_labels = st.data_editor(
            label_rows,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "name": st.column_config.TextColumn("Channel", disabled=True),
                "canonical_label": st.column_config.SelectboxColumn(
                    "Canonical slot", options=[""] + reference_labels, required=True
                ),
            },
            key=f"canonical-label-editor-{selected_label}",
        )
        label_records = edited_labels.to_dict("records") if hasattr(edited_labels, "to_dict") else edited_labels
        canonical_labels = tuple(str(row["canonical_label"]) for row in label_records)

    if st.button("Run alignment", type="primary"):
        try:
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("Channel names must be non-empty and unique")
            if canonical_labels is None and not np.isfinite(positions).all():
                raise ValueError("All edited coordinates must be finite")
            if canonical_labels is not None and any(not label for label in canonical_labels):
                raise ValueError("Choose a canonical slot for every row")
            optodes = montage.optodes
            if optodes is not None and axis_selections is not None:
                optodes = OptodeGeometry(
                    optodes.source_names,
                    _apply_axis_selections(optodes.source_positions, axis_selections),
                    optodes.detector_names,
                    _apply_axis_selections(optodes.detector_positions, axis_selections),
                    optodes.channel_pairs,
                )
            adjusted_montage = MontageInput(
                names,
                positions,
                unit,
                frame,
                montage.modality,
                optodes,
            )
            result = align(
                adjusted_montage,
                reference=reference,
                canonical_labels=canonical_labels,
            )
            results = dict(st.session_state.get("alignment_results", {}))
            results[selected_label] = result
            st.session_state.alignment_results = results
            aligned_montages = dict(st.session_state.get("alignment_montages", {}))
            aligned_montages[selected_label] = adjusted_montage
            st.session_state.alignment_montages = aligned_montages
            st.session_state.alignment_result = result
            st.success(f"Mapped {int(result.valid.sum())}/{len(result.valid)} inputs with {result.method}")
        except Exception as error:
            st.error(str(error))

    selected_result = st.session_state.get("alignment_results", {}).get(selected_label)
    if selected_result is not None:
        st.subheader("2D alignment preview")
        aligned_montage = st.session_state.get("alignment_montages", {}).get(selected_label)
        optodes_available = aligned_montage is not None and aligned_montage.optodes is not None
        layer_options = ["1005", "Midpoints", "Assigned", "Assignment lines"]
        if optodes_available:
            layer_options.extend(["S-D lines", "Sources", "Detectors"])
        visible_layers = st.segmented_control(
            "2D layers",
            layer_options,
            default=["1005", "Midpoints", "Assigned", "Assignment lines", "S-D lines"]
            if optodes_available
            else ["1005", "Midpoints", "Assigned", "Assignment lines"],
            selection_mode="multi",
            key=f"2d-layers-v2-{selected_label}",
        ) or []
        show_reference = "1005" in visible_layers
        show_midpoints = "Midpoints" in visible_layers
        show_assigned_slots = "Assigned" in visible_layers
        show_assignment_lines = "Assignment lines" in visible_layers
        show_sd_lines = "S-D lines" in visible_layers
        show_sources = "Sources" in visible_layers
        show_detectors = "Detectors" in visible_layers
        if not optodes_available:
            show_sources = show_detectors = show_sd_lines = False
        else:
            st.caption("Source and detector markers are off by default; select their layers when needed.")
        st.plotly_chart(
            _alignment_2d_figure(
                selected_result,
                montage=aligned_montage,
                show_reference=show_reference,
                show_midpoints=show_midpoints,
                show_assigned_slots=show_assigned_slots,
                show_assignment_lines=show_assignment_lines,
                show_sources=show_sources,
                show_detectors=show_detectors,
                show_sd_lines=show_sd_lines,
            ),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
        )
        with st.expander("2D aligned coordinate table"):
            st.caption(
                "Midpoint x/y is the aligned channel location before snapping; assigned slot x/y is the "
                "selected Standard position. Distance is computed in 3D for 3D inputs."
            )
            st.dataframe(
                _alignment_2d_rows(selected_result),
                hide_index=True,
                width="stretch",
            )


def _adjust_and_map(st) -> None:
    mode = st.segmented_control(
        "Workflow",
        ["Imported geometry", "Manual S-D placement"],
        default="Imported geometry",
        selection_mode="single",
    )
    if mode == "Manual S-D placement":
        st.subheader("Standard 1005 head map")
        st.caption("Place sources and detectors; accepted S-D pairs become the current alignment result.")
        st.fragment(_interactive_map)(st)
    else:
        _adjust_imported_geometry(st)


def _standard_scalp_mesh(
    reference_positions: np.ndarray,
    *,
    clearance_m: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    positions = np.asarray(reference_positions, dtype=np.float64)
    design = np.column_stack([2.0 * positions, np.ones(len(positions))])
    sphere_target = np.square(positions).sum(axis=1)
    center = np.linalg.lstsq(design, sphere_target, rcond=None)[0][:3]
    radial = positions - center
    radii = np.linalg.norm(radial, axis=1, keepdims=True)
    expanded = positions + radial / radii * clearance_m
    hull = ConvexHull(expanded)
    return expanded, hull.simplices, center


def _interactive_3d_figure(result: AlignmentResult, *, camera_scale: float = 1.0):
    import plotly.graph_objects as go

    reference = load_reference(result.reference)
    labels = np.asarray(reference["labels"])
    reference_positions = np.asarray(reference["positions"])
    lower_bounds = np.nanmin(reference_positions, axis=0)
    upper_bounds = np.nanmax(reference_positions, axis=0)
    center = (lower_bounds + upper_bounds) / 2.0
    extents = (upper_bounds - lower_bounds) / 2.0
    scalp_vertices, scalp_triangles, _ = _standard_scalp_mesh(reference_positions)
    figure = go.Figure()
    figure.add_trace(go.Mesh3d(
        x=scalp_vertices[:, 0],
        y=scalp_vertices[:, 1],
        z=scalp_vertices[:, 2],
        i=scalp_triangles[:, 0],
        j=scalp_triangles[:, 1],
        k=scalp_triangles[:, 2],
        opacity=0.32,
        color="#b9bfbc",
        lighting={"ambient": 0.72, "diffuse": 0.72, "roughness": 0.88, "specular": 0.08},
        lightposition={"x": 120, "y": 180, "z": 220},
        hoverinfo="skip",
        name="Scalp",
    ))
    front_point = scalp_vertices[int(np.argmax(scalp_vertices[:, 1]))]
    nose_base_y = front_point[1] - 0.006
    nose_tip_y = front_point[1] + 0.020
    figure.add_trace(
        go.Mesh3d(
            x=[
                front_point[0] - 0.008,
                front_point[0] + 0.008,
                front_point[0] - 0.005,
                front_point[0] + 0.005,
                front_point[0],
            ],
            y=[nose_base_y, nose_base_y, nose_base_y, nose_base_y, nose_tip_y],
            z=[
                front_point[2] - 0.008,
                front_point[2] - 0.008,
                front_point[2] + 0.011,
                front_point[2] + 0.011,
                front_point[2] - 0.001,
            ],
            i=[0, 0, 1, 2, 0, 1],
            j=[1, 2, 3, 3, 4, 4],
            k=[2, 1, 2, 4, 2, 3],
            color="#8f9693",
            opacity=0.78,
            flatshading=True,
            hovertemplate="ANTERIOR (+y)<extra>Nose</extra>",
            name="Nose (front)",
            showlegend=False,
        )
    )
    orientation_axes = (
        (0, "#d32f2f", "LEFT (-x)", "RIGHT (+x)", "x direction"),
        (1, "#2e7d32", "POSTERIOR (-y)", "ANTERIOR (+y)", "y direction"),
        (2, "#1565c0", "INFERIOR (-z)", "SUPERIOR (+z)", "z direction"),
    )
    for axis, color, negative_label, positive_label, trace_name in orientation_axes:
        negative = center.copy()
        positive = center.copy()
        negative_scale = 1.36 if axis == 1 else 1.22
        positive_scale = 1.48 if axis == 1 else 1.22
        negative[axis] -= extents[axis] * negative_scale
        positive[axis] += extents[axis] * positive_scale
        figure.add_trace(
            go.Scatter3d(
                x=[negative[0], center[0], positive[0]],
                y=[negative[1], center[1], positive[1]],
                z=[negative[2], center[2], positive[2]],
                mode="lines+markers+text",
                line={"color": color, "width": 9},
                marker={"color": color, "size": [5, 4, 8], "symbol": "diamond"},
                text=[negative_label, "", positive_label],
                textposition="top center",
                textfont={"size": 14, "color": color},
                hovertemplate="%{text}<extra>" + trace_name + "</extra>",
                name=trace_name,
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=reference_positions[:, 0],
            y=reference_positions[:, 1],
            z=reference_positions[:, 2],
            mode="markers",
            marker={"size": 3.5, "color": "#9aa49f", "opacity": 0.55},
            text=labels,
            hovertemplate="%{text}<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m<extra>Standard 1005</extra>",
            name="Standard 1005",
        )
    )
    assigned = result.ch_pos_1005[result.valid]
    assigned_labels = np.asarray(result.slot_labels)[result.valid]
    collision_ids = set(result.collisions)
    collision_mask = np.asarray(
        [int(slot_id) in collision_ids for slot_id in result.slot_ids[result.valid]],
        dtype=bool,
    )
    for mask, color, trace_name in (
        (~collision_mask, "#f2b51d", "Assigned slots"),
        (collision_mask, "#a65300", "Assigned slots (collision)"),
    ):
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter3d(
                x=assigned[mask, 0],
                y=assigned[mask, 1],
                z=assigned[mask, 2],
                mode="markers",
                marker={"size": 8, "color": color, "line": {"color": "#3e3320", "width": 1.5}},
                text=assigned_labels[mask],
                hovertemplate=(
                    "%{text}<br>x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m"
                    f"<extra>{trace_name}</extra>"
                ),
                name=trace_name,
            )
        )
    figure.update_layout(
        height=680,
        margin={"l": 55, "r": 55, "t": 10, "b": 75},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        dragmode="turntable",
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "aspectmode": "data",
            "dragmode": "turntable",
            "xaxis": {
                "title": "x (m)",
                "range": [center[0] - extents[0] * 1.32, center[0] + extents[0] * 1.32],
                "showbackground": False,
                "gridcolor": "#f3c5c5",
                "color": "#a82222",
                "zeroline": False,
                "showline": False,
                "showspikes": False,
                "tickformat": ".02f",
            },
            "yaxis": {
                "title": "y (m)",
                "range": [center[1] - extents[1] * 1.48, center[1] + extents[1] * 1.65],
                "showbackground": False,
                "gridcolor": "#c8e2ca",
                "color": "#236b28",
                "zeroline": False,
                "showline": False,
                "showspikes": False,
                "tickformat": ".02f",
            },
            "zaxis": {
                "title": "z (m)",
                "range": [center[2] - extents[2] * 1.22, center[2] + extents[2] * 1.28],
                "showbackground": False,
                "gridcolor": "#c5d8ef",
                "color": "#124f96",
                "zeroline": False,
                "showline": False,
                "showspikes": False,
                "tickformat": ".02f",
            },
            "camera": {
                "eye": {
                    "x": 2.10 * camera_scale,
                    "y": 0.70 * camera_scale,
                    "z": 0.74 * camera_scale,
                },
                "projection": {"type": "perspective"},
            },
        },
        legend={"orientation": "h", "y": 0.02, "x": 0.02, "font": {"size": 15}, "itemsizing": "constant"},
    )
    return figure


def _add_optodes_to_3d_figure(figure, montage: MontageInput, result: AlignmentResult):
    import plotly.graph_objects as go

    if montage.optodes is None:
        return figure
    optodes = montage.optodes
    sources = transform_positions(montage, optodes.source_positions, reference=result.reference)
    detectors = transform_positions(montage, optodes.detector_positions, reference=result.reference)
    line_x, line_y, line_z = [], [], []
    for source_index, detector_index in optodes.channel_pairs:
        source = sources[source_index]
        detector = detectors[detector_index]
        line_x.extend([source[0], detector[0], None])
        line_y.extend([source[1], detector[1], None])
        line_z.extend([source[2], detector[2], None])
    figure.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": "#5c6662", "width": 5},
            hoverinfo="skip",
            name="S-D channels",
        )
    )
    for positions, names, color, trace_name in (
        (sources, optodes.source_names, "#e53935", "Sources"),
        (detectors, optodes.detector_names, "#1976d2", "Detectors"),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers+text",
                marker={"size": 9, "color": color, "line": {"color": "white", "width": 1.5}},
                text=names,
                textposition="top center",
                hovertemplate="%{text}<extra>" + trace_name + "</extra>",
                name=trace_name,
            )
        )
    return figure


def _plotly_review_config() -> dict[str, object]:
    return {
        "displaylogo": False,
        "scrollZoom": False,
        "modeBarButtonsToRemove": ["zoom3d", "pan3d", "orbitRotation"],
    }


def _export_payloads(result: AlignmentResult) -> tuple[str, str, bytes]:
    tsv_buffer = io.StringIO()
    writer = csv.writer(tsv_buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(
        ["name", "slot_id", "slot_label", "x", "y", "z", "valid", "distance", "distance_unit"]
    )
    for index, name in enumerate(result.names):
        writer.writerow(
            [
                name,
                int(result.slot_ids[index]),
                result.slot_labels[index],
                *result.ch_pos_1005[index].tolist(),
                bool(result.valid[index]),
                float(result.distances[index]),
                result.distance_unit,
            ]
        )

    def json_safe(value):
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    payload = {
        "ch_names": list(result.names),
        "ch_slot_1005_id": result.slot_ids.tolist(),
        "ch_slot_1005_label": list(result.slot_labels),
        "ch_pos_1005": json_safe(result.ch_pos_1005),
        "ch_pos_1005_valid": result.valid.tolist(),
        "ch_pos_1005_distance": json_safe(result.distances),
        "collisions": {str(slot): list(names) for slot, names in result.collisions.items()},
        "meta_ch_pos_1005_distance_unit": result.distance_unit,
        "meta_ch_pos_1005_frame": result.reference,
        "meta_alignment_method": result.method,
    }
    npz_buffer = io.BytesIO()
    np.savez_compressed(
        npz_buffer,
        ch_names=np.asarray(result.names),
        ch_slot_1005_id=result.slot_ids,
        ch_slot_1005_label=np.asarray(result.slot_labels),
        ch_pos_1005=result.ch_pos_1005.astype(np.float32),
        ch_pos_1005_valid=result.valid,
        ch_pos_1005_distance=result.distances.astype(np.float32),
        meta_ch_pos_1005_distance_unit=np.asarray(result.distance_unit),
        meta_ch_pos_1005_frame=np.asarray(result.reference),
        meta_alignment_method=np.asarray(result.method),
    )
    return tsv_buffer.getvalue(), json.dumps(payload, indent=2) + "\n", npz_buffer.getvalue()


def _mtrenirs_sidecar_npz(result: AlignmentResult, channel_order: str) -> bytes:
    if channel_order not in {"pair-interleaved", "chromophore-grouped"}:
        raise ValueError("channel_order must be 'pair-interleaved' or 'chromophore-grouped'")

    rows: list[tuple[str, str, np.ndarray, bool]] = []
    for name, position, valid in zip(result.names, result.ch_pos_1005, result.valid):
        parts = str(name).rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].lower() in {"hbo", "hbr"}:
            rows.append((parts[0], parts[1].lower(), position, bool(valid)))
        else:
            rows.extend(
                (
                    (str(name), "hbo", position, bool(valid)),
                    (str(name), "hbr", position, bool(valid)),
                )
            )

    pair_order = {pair: index for index, pair in enumerate(dict.fromkeys(row[0] for row in rows))}
    kind_order = {"hbo": 0, "hbr": 1}
    if channel_order == "pair-interleaved":
        rows.sort(key=lambda row: (pair_order[row[0]], kind_order[row[1]]))
    else:
        rows.sort(key=lambda row: (kind_order[row[1]], pair_order[row[0]]))

    output = io.BytesIO()
    np.savez_compressed(
        output,
        ch_names=np.asarray([f"{pair} {kind}" for pair, kind, _, _ in rows]),
        ch_types=np.asarray([kind for _, kind, _, _ in rows]),
        ch_pos_1005=np.asarray([position for _, _, position, _ in rows], dtype=np.float32),
        ch_pos_1005_valid=np.asarray([valid for _, _, _, valid in rows], dtype=bool),
        meta_ch_pos_1005_frame=np.asarray(result.reference),
        meta_channel_order=np.asarray(channel_order),
        meta_mtrenirs_role=np.asarray("geometry-sidecar"),
    )
    return output.getvalue()


def _batch_export_zip(results: dict[str, AlignmentResult], formats: list[str]) -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, result in results.items():
            safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in label)
            tsv_data, json_data, npz_data = _export_payloads(result)
            payloads = {"TSV": tsv_data, "JSON": json_data, "NPZ": npz_data}
            for format_name in formats:
                archive.writestr(
                    f"{safe_label}/mapping.{format_name.lower()}",
                    payloads[format_name],
                )
    return archive_buffer.getvalue()


def _review_and_export(st) -> None:
    results = dict(st.session_state.get("alignment_results", {}))
    current = st.session_state.get("alignment_result")
    if not results and current is not None:
        results["manual_sd"] = current
    if not results:
        st.info("Run an imported or manual alignment in the second tab.")
        return
    selected_label = st.selectbox("Aligned montage", list(results), key="review-result")
    result = results[selected_label]
    st.session_state.alignment_result = result
    selected_entry = next(
        (
            entry
            for entry in st.session_state.get("imported_batch", [])
            if entry["label"] == selected_label
        ),
        None,
    )
    if selected_entry is not None:
        represented_files = selected_entry.get("source_files", [selected_entry["label"]])
        st.caption(
            f"Showing geometry shared by {len(represented_files)} input file"
            + ("." if len(represented_files) == 1 else "s.")
        )
        with st.expander("Represented patient / recording files"):
            st.code("\n".join(represented_files))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Inputs", len(result.names))
    metric_columns[1].metric("Valid", int(result.valid.sum()))
    metric_columns[2].metric("Slots", int(np.unique(result.slot_ids[result.valid]).size))
    metric_columns[3].metric("Collisions", len(result.collisions))
    aligned_montage = st.session_state.get("alignment_montages", {}).get(selected_label)
    optodes_available = aligned_montage is not None and aligned_montage.optodes is not None
    st.caption(
        "3D colors: yellow = assigned slot; dark amber = collision; with optodes shown, "
        "red = source and blue = detector. The red axis points right (+x)."
    )
    show_optodes = st.toggle(
        "Show S-D optodes (red sources, blue detectors)",
        value=False,
        disabled=not optodes_available,
        key=f"show-optodes-{selected_label}",
    )
    zoom_key = f"review-3d-zoom-{selected_label}"
    zoom_level = int(st.session_state.get(zoom_key, 0))
    zoom_spacer, zoom_out_column, zoom_in_column = st.columns([12, 1, 1])
    if zoom_out_column.button(
        "−",
        key=f"review-3d-zoom-out-{selected_label}",
        help="Zoom out",
        disabled=zoom_level <= 0,
        width="stretch",
    ):
        zoom_level -= 1
        st.session_state[zoom_key] = zoom_level
    if zoom_in_column.button(
        "+",
        key=f"review-3d-zoom-in-{selected_label}",
        help="Zoom in",
        disabled=zoom_level >= 4,
        width="stretch",
    ):
        zoom_level += 1
        st.session_state[zoom_key] = zoom_level
    figure = _interactive_3d_figure(result, camera_scale=1.0 - zoom_level * 0.1)
    if show_optodes:
        figure = _add_optodes_to_3d_figure(figure, aligned_montage, result)
    st.plotly_chart(
        figure,
        width="stretch",
        config=_plotly_review_config(),
    )
    tsv_data, json_data, npz_data = _export_payloads(result)
    download_columns = st.columns(3)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in selected_label
    )
    download_columns[0].download_button("Download TSV", tsv_data, f"{safe_label}_mapping.tsv", "text/tab-separated-values")
    download_columns[1].download_button("Download JSON", json_data, f"{safe_label}_mapping.json", "application/json")
    download_columns[2].download_button("Download NPZ", npz_data, f"{safe_label}_mapping.npz", "application/octet-stream")
    st.subheader("MtReNIRS geometry sidecar")
    st.markdown(
        '<p style="font-size:0.85rem;color:#6b7280">'
        'Geometry only: combine this file with preprocessed 10 Hz, run-normalized HbO/HbR signals. '
        '<span style="text-decoration:underline;text-underline-offset:0.15em">'
        'Signal channels must be in exactly the same order as this exported sidecar.'
        '</span> This sidecar is not a standalone signal input.'
        '</p>',
        unsafe_allow_html=True,
    )
    order_label = st.segmented_control(
        "Channel order",
        ["Pair-interleaved", "Chromophore-grouped"],
        default="Pair-interleaved",
        key=f"mtrenirs-order-{safe_label}",
    )
    channel_order = {
        "Pair-interleaved": "pair-interleaved",
        "Chromophore-grouped": "chromophore-grouped",
    }[order_label]
    if channel_order == "pair-interleaved":
        order_example = (
            'S1-D1 <strong style="color:#111111">HbO</strong>, '
            'S1-D1 <strong style="color:#111111">HbR</strong>, '
            'S2-D1 <strong style="color:#111111">HbO</strong>, '
            'S2-D1 <strong style="color:#111111">HbR</strong>, …'
        )
    else:
        order_example = (
            'S1-D1 <strong style="color:#111111">HbO</strong>, '
            'S2-D1 <strong style="color:#111111">HbO</strong>, …, '
            'S1-D1 <strong style="color:#111111">HbR</strong>, '
            'S2-D1 <strong style="color:#111111">HbR</strong>, …'
        )
    st.markdown(
        '<p style="font-size:0.82rem;color:#6b7280;margin:0.1rem 0 0.5rem">'
        f'Example: {order_example}</p>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download MtReNIRS sidecar NPZ",
        _mtrenirs_sidecar_npz(result, channel_order),
        f"{safe_label}_mtrenirs_sidecar.npz",
        "application/octet-stream",
    )
    if len(results) > 1:
        formats = st.multiselect("Batch export formats", ["TSV", "JSON", "NPZ"], default=["TSV", "JSON", "NPZ"])
        st.download_button(
            f"Download {len(results)} montages as ZIP",
            _batch_export_zip(results, formats),
            "montage_alignment_batch.zip",
            "application/zip",
            disabled=not formats,
        )


def _interactive_map(st: object) -> None:
    reference = load_reference()
    labels = reference["labels"]
    points = np.asarray(reference["positions_2d"])
    xyz = np.asarray(reference["positions"])
    standard_1020_labels = [label for label in _CLASSIC_1020_LABELS if label in labels]

    if "sources" not in st.session_state:
        st.session_state.sources = {}
        st.session_state.detectors = {}
        st.session_state.pairs = []
        st.session_state.editor_revision = 0

    placement_mode = st.segmented_control(
        "Placement mode",
        options=["S", "D"],
        default="S",
        selection_mode="single",
    )
    min_distance_column, max_distance_column = st.columns(2)
    with min_distance_column:
        min_distance_mm = st.number_input(
            "Min S-D distance (mm)",
            min_value=0.0,
            value=_DEFAULT_SD_MIN_MM,
            step=1.0,
        )
    with max_distance_column:
        max_distance_mm = st.number_input(
            "Max S-D distance (mm)",
            min_value=0.0,
            value=_DEFAULT_SD_MAX_MM,
            step=1.0,
        )
    source_toggle, detector_toggle, line_toggle, midpoint_toggle = st.columns(4)
    with source_toggle:
        show_sources = st.toggle("Sources", value=True)
    with detector_toggle:
        show_detectors = st.toggle("Detectors", value=True)
    with line_toggle:
        show_lines = st.toggle("S-D lines", value=True)
    with midpoint_toggle:
        show_midpoints = st.toggle("Midpoints", value=True)
    if st.button("Reset all", type="secondary", disabled=not (
        st.session_state.sources or st.session_state.detectors or st.session_state.pairs
    )):
        st.session_state.sources = {}
        st.session_state.detectors = {}
        st.session_state.pairs = []
        st.session_state.editor_revision += 1
        st.rerun(scope="fragment")

    placed_by_label = {
        label: ("S", name) for name, label in st.session_state.sources.items()
    } | {
        label: ("D", name) for name, label in st.session_state.detectors.items()
    }
    point_by_label = {label: points[index] for index, label in enumerate(labels)}
    xyz_by_label = {label: xyz[index] for index, label in enumerate(labels)}
    pair_distances_mm = {
        (source, detector): float(
            np.linalg.norm(
                xyz_by_label[source_label] - xyz_by_label[detector_label]
            ) * 1000
        )
        for source, source_label in st.session_state.sources.items()
        for detector, detector_label in st.session_state.detectors.items()
    }
    if min_distance_mm > max_distance_mm:
        st.error("Minimum S-D distance must not exceed the maximum.")
        st.session_state.pairs = []
    else:
        st.session_state.pairs = [
            pair
            for pair, distance_mm in pair_distances_mm.items()
            if min_distance_mm <= distance_mm <= max_distance_mm
        ]
    rejected_pair_count = len(pair_distances_mm) - len(st.session_state.pairs)
    if pair_distances_mm:
        st.caption(
            f"{len(st.session_state.pairs)} S-D pair(s) within "
            f"[{min_distance_mm:g}, {max_distance_mm:g}] mm; "
            f"{rejected_pair_count} rejected."
        )
    head_radius = float(np.nanmax(np.linalg.norm(points, axis=1)))

    def map_point(label: str, name: str | None = None) -> dict[str, object]:
        point = point_by_label[label] / head_radius
        output: dict[str, object] = {
            "label": label,
            "x": float(point[0]),
            "y": float(-point[1]),
        }
        if name is not None:
            output["name"] = name
        return output

    component = _get_head_map_component()
    result = component(
        data={
            "points": [map_point(label) for label in labels],
            "landmarks": [map_point(label) for label in standard_1020_labels],
            "sources": [map_point(label, name) for name, label in st.session_state.sources.items()],
            "detectors": [map_point(label, name) for name, label in st.session_state.detectors.items()],
            "pairs": [
                {"source": source, "detector": detector}
                for source, detector in st.session_state.pairs
            ],
            "showSources": show_sources,
            "showDetectors": show_detectors,
            "showLines": show_lines,
            "showMidpoints": show_midpoints,
        },
        key="head-map",
        on_clicked_change=lambda: None,
    )
    clicked = result.clicked
    if clicked:
        clicked_label = clicked["label"]
        existing = placed_by_label.get(clicked_label)
        if existing:
            kind, name = existing
            collection = st.session_state.sources if kind == "S" else st.session_state.detectors
            del collection[name]
            st.session_state.pairs = [pair for pair in st.session_state.pairs if name not in pair]
        else:
            collection = (
                st.session_state.sources if placement_mode == "S" else st.session_state.detectors
            )
            prefix = placement_mode or "S"
            used_numbers = {
                int(name[1:]) for name in collection if name[1:].isdigit()
            }
            number = next(index for index in range(1, len(collection) + 2) if index not in used_numbers)
            collection[f"{prefix}{number}"] = clicked_label
        st.session_state.editor_revision += 1
        st.rerun(scope="fragment")

    placed_rows = [
        {"optode": name, "type": "Source", "slot": label}
        for name, label in st.session_state.sources.items()
    ] + [
        {"optode": name, "type": "Detector", "slot": label}
        for name, label in st.session_state.detectors.items()
    ]
    if placed_rows:
        edited = st.data_editor(
            placed_rows,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "optode": st.column_config.TextColumn("Optode name", required=True),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Source", "Detector"],
                    required=True,
                ),
                "slot": st.column_config.SelectboxColumn(
                    "Standard 1005 slot",
                    options=list(labels),
                    required=True,
                ),
            },
            key=f"optode-editor-{st.session_state.editor_revision}",
        )
        edited_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
        if edited_rows != placed_rows:
            names = [str(row["optode"]).strip() for row in edited_rows]
            types = [str(row["type"]) for row in edited_rows]
            slots = [str(row["slot"]) for row in edited_rows]
            validation_error = None
            if any(not name for name in names):
                validation_error = "Optode names cannot be empty."
            elif len(names) != len(set(names)):
                validation_error = "Optode names must be unique."
            elif len(slots) != len(set(slots)):
                validation_error = "Each Standard 1005 slot can contain only one optode."
            elif any(kind not in {"Source", "Detector"} for kind in types):
                validation_error = "Choose Source or Detector for every optode."
            elif any(slot not in point_by_label for slot in slots):
                validation_error = "Choose a valid Standard 1005 slot."

            if validation_error:
                st.error(validation_error)
            else:
                source_renames: dict[str, str] = {}
                detector_renames: dict[str, str] = {}
                new_sources: dict[str, str] = {}
                new_detectors: dict[str, str] = {}
                for original, kind, name, slot in zip(placed_rows, types, names, slots):
                    if original["type"] == "Source" and kind == "Source":
                        source_renames[original["optode"]] = name
                    elif original["type"] == "Detector" and kind == "Detector":
                        detector_renames[original["optode"]] = name

                    if kind == "Source":
                        new_sources[name] = slot
                    else:
                        new_detectors[name] = slot
                st.session_state.sources = new_sources
                st.session_state.detectors = new_detectors
                st.session_state.pairs = [
                    (source_renames[source], detector_renames[detector])
                    for source, detector in st.session_state.pairs
                    if source in source_renames and detector in detector_renames
                ]
                st.session_state.editor_revision += 1
                st.rerun(scope="fragment")

    source_names = list(st.session_state.sources)
    detector_names = list(st.session_state.detectors)
    if source_names and detector_names:
        lookup = {label: index for index, label in enumerate(labels)}
        source_xyz = np.asarray([xyz[lookup[label]] for label in st.session_state.sources.values()])
        detector_xyz = np.asarray([xyz[lookup[label]] for label in st.session_state.detectors.values()])
        if st.session_state.pairs:
            result = align_optodes(
                source_positions=source_xyz,
                detector_positions=detector_xyz,
                channel_pairs=[
                    (source_names.index(source), detector_names.index(detector))
                    for source, detector in st.session_state.pairs
                ],
                channel_names=[f"{source}-{detector}" for source, detector in st.session_state.pairs],
                frame="standard_1005",
            )
            sd_distances = [pair_distances_mm[pair] for pair in st.session_state.pairs]
            st.dataframe(
                {
                    "channel": result.names,
                    "slot": result.slot_labels,
                    "sd_distance_mm": np.round(sd_distances, 1),
                }
            )
            manual_signature = (
                tuple(st.session_state.sources.items()),
                tuple(st.session_state.detectors.items()),
                tuple(st.session_state.pairs),
            )
            if st.button("Run manual alignment", type="primary"):
                channel_pairs = tuple(
                    (source_names.index(source), detector_names.index(detector))
                    for source, detector in st.session_state.pairs
                )
                manual_montage = MontageInput(
                    result.names,
                    result.source_positions,
                    "m",
                    "standard_1005",
                    "fnirs",
                    OptodeGeometry(
                        tuple(source_names),
                        source_xyz,
                        tuple(detector_names),
                        detector_xyz,
                        channel_pairs,
                    ),
                )
                st.session_state.alignment_result = result
                manual_results = dict(st.session_state.get("alignment_results", {}))
                manual_results["manual_sd"] = result
                st.session_state.alignment_results = manual_results
                aligned_montages = dict(st.session_state.get("alignment_montages", {}))
                aligned_montages["manual_sd"] = manual_montage
                st.session_state.alignment_montages = aligned_montages
                st.session_state.manual_alignment_signature = manual_signature
            saved_manual = st.session_state.get("alignment_results", {}).get("manual_sd")
            if (
                saved_manual is not None
                and st.session_state.get("manual_alignment_signature") == manual_signature
            ):
                st.success("Manual alignment is ready. Review and download it in 3. 3D Review & Export.")
                manual_layers = st.segmented_control(
                    "2D layers",
                    [
                        "1005",
                        "Midpoints",
                        "Assigned",
                        "Assignment lines",
                        "S-D lines",
                        "Sources",
                        "Detectors",
                    ],
                    default=["1005", "Midpoints", "Assigned", "Assignment lines", "S-D lines"],
                    selection_mode="multi",
                    key="manual-2d-layers",
                ) or []
                manual_montage = st.session_state.get("alignment_montages", {}).get("manual_sd")
                st.plotly_chart(
                    _alignment_2d_figure(
                        saved_manual,
                        montage=manual_montage,
                        show_reference="1005" in manual_layers,
                        show_midpoints="Midpoints" in manual_layers,
                        show_assigned_slots="Assigned" in manual_layers,
                        show_assignment_lines="Assignment lines" in manual_layers,
                        show_sources="Sources" in manual_layers,
                        show_detectors="Detectors" in manual_layers,
                        show_sd_lines="S-D lines" in manual_layers,
                    ),
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False},
                    key="manual-alignment-2d-result",
                )


def main() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as error:
        raise SystemExit('Install montage-align with: pip install -e .') from error

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    app()
