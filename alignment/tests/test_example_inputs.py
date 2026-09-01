# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtrenirs.alignment import align
from mtrenirs.alignment.io import (
    load_bids_nirs_sidecars,
    load_json,
    load_npz,
    load_position_text,
    load_positions_tsv,
    load_snirf,
)
from mtrenirs.alignment.ui import _discover_input_paths, _scan_path_batch


ROOT = Path(__file__).resolve().parents[1] / "example_test_input"


def _assert_expected(scenario: str, montage, canonical_labels=None) -> None:
    result = align(montage, canonical_labels=canonical_labels)
    expected = json.loads((ROOT / scenario / "expected_alignment.json").read_text(encoding="utf-8"))

    assert result.method == expected["method"]
    assert len(result.names) == expected["rows"]
    assert int(result.valid.sum()) == expected["valid_rows"]
    assert list(result.slot_labels) == expected["slot_labels"]
    assert result.distance_unit == expected["distance_unit"]
    assert len(result.collisions) == len(expected["collision_slots"])


def test_public_examples_are_synthetic_and_have_no_symlinks() -> None:
    assert (ROOT / "README.md").is_file()
    assert not any(path.is_symlink() for path in ROOT.rglob("*"))
    assert {path.name for path in ROOT.iterdir() if path.is_dir()} == {
        "01-canonical-labels",
        "02-positions-3d",
        "03-positions-2d",
        "04-standard-frame",
        "05-missing-position",
        "06-bids-fnirs",
        "07-minimal-snirf",
        "08-canonical-collision",
    }


def test_ui_scanner_loads_each_scenario_once() -> None:
    entries, report = _scan_path_batch(
        _discover_input_paths(ROOT),
        unit="m",
        frame="head",
        modality="fNIRS",
    )

    assert len(entries) == 8
    assert all(8 <= len(entry["montage"].names) <= 12 for entry in entries)
    assert sum(row["status"] == "Accepted" for row in report) == 8
    assert sum(row["status"] == "Skipped" for row in report) == 8
    assert not [row for row in report if row["status"] == "Warning"]


def test_canonical_and_coordinate_examples_match_expected_results() -> None:
    canonical_path = ROOT / "01-canonical-labels/canonical_labels.json"
    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    _assert_expected(
        "01-canonical-labels",
        load_json(canonical_path),
        tuple(canonical_payload["canonical_labels"]),
    )
    _assert_expected("02-positions-3d", load_positions_tsv(ROOT / "02-positions-3d/positions_3d.tsv"))
    _assert_expected("03-positions-2d", load_position_text(ROOT / "03-positions-2d/positions_2d.txt"))
    _assert_expected("04-standard-frame", load_npz(ROOT / "04-standard-frame/standard_frame.npz"))
    missing = load_json(ROOT / "05-missing-position/missing_position.json")
    assert np.isnan(missing.positions[missing.names.index("missing")]).all()
    _assert_expected("05-missing-position", missing)


def test_fnirs_examples_preserve_physical_pairs() -> None:
    bids_root = ROOT / "06-bids-fnirs"
    bids = load_bids_nirs_sidecars(
        bids_root / "sub-synthetic_optodes.tsv",
        bids_root / "sub-synthetic_coordsystem.json",
        bids_root / "sub-synthetic_channels.tsv",
        min_sd_distance_mm=8.0,
        max_sd_distance_mm=35.0,
    )
    assert bids.names == tuple(f"S{index}-D{index}" for index in range(1, 9))
    assert bids.optodes is not None
    _assert_expected("06-bids-fnirs", bids)

    snirf = load_snirf(
        ROOT / "07-minimal-snirf/synthetic_recording.snirf",
        physical_pairs=True,
        min_sd_distance_mm=8.0,
        max_sd_distance_mm=35.0,
    )
    assert snirf.names == tuple(f"S{index}-D{index}" for index in range(1, 9))
    assert snirf.optodes is not None
    _assert_expected("07-minimal-snirf", snirf)


def test_collision_example_reports_two_collisions() -> None:
    path = ROOT / "08-canonical-collision/canonical_collision.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    montage = load_json(path)
    result = align(montage, canonical_labels=tuple(payload["canonical_labels"]))

    _assert_expected("08-canonical-collision", montage, tuple(payload["canonical_labels"]))
    assert len(result.collisions) == 2
    assert set(result.collisions.values()) == {("left-a", "left-b"), ("back-a", "back-b")}