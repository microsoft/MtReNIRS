# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import io
import zipfile

import numpy as np

from mtrenirs.alignment.alignment import align, load_reference
from mtrenirs.alignment.models import MontageInput, OptodeGeometry
from mtrenirs.alignment.ui import (
    _alignment_2d_figure,
    _alignment_2d_rows,
    _batch_export_zip,
    _discover_input_paths,
    _export_payloads,
    _mtrenirs_sidecar_npz,
    _interactive_3d_figure,
    _load_geometry_path,
    _load_path_batch,
    _load_uploaded_batch,
    _missing_coordinate_inputs,
    _parsed_geometry_preview_rows,
    _plotly_review_config,
    _scan_path_batch,
    _standard_scalp_mesh,
)


def test_parsed_geometry_preview_preserves_canonical_coordinate_placeholders():
    montage = MontageInput(
        names=("row-1", "row-2"),
        positions=np.zeros((2, 3)),
    )

    canonical_rows = _parsed_geometry_preview_rows(montage, ("Fp1", "Cz"))
    assert canonical_rows == [
        {"name": "row-1", "canonical_label": "Fp1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"name": "row-2", "canonical_label": "Cz", "x": 0.0, "y": 0.0, "z": 0.0},
    ]

    geometry_rows = _parsed_geometry_preview_rows(montage, None)
    assert geometry_rows[0] == {"name": "row-1", "x": 0.0, "y": 0.0, "z": 0.0}


def test_missing_coordinate_inputs_excludes_canonical_placeholders():
    missing = MontageInput(
        names=("valid", "missing"),
        positions=np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]),
    )
    canonical = MontageInput(
        names=("canonical",),
        positions=np.zeros((1, 3)),
    )

    detected = _missing_coordinate_inputs(
        [
            {"label": "geometry", "montage": missing, "canonical_labels": None},
            {"label": "labels", "montage": canonical, "canonical_labels": ("Cz",)},
        ]
    )

    assert detected == {"geometry": ("missing",)}


def test_ui_exports_and_3d_figure_are_complete():
    result = align(
        positions=np.asarray([[0.0, 0.08, 0.08], [0.03, 0.07, 0.07]]),
        names=("S1-D1", "S2-D1"),
        frame="head",
    )

    tsv_data, json_data, npz_data = _export_payloads(result)

    assert "distance_unit" in tsv_data.splitlines()[0]
    assert json.loads(json_data)["meta_alignment_method"] == result.method
    with np.load(__import__("io").BytesIO(npz_data), allow_pickle=False) as exported:
        assert exported["ch_pos_1005"].shape == (2, 3)
        assert str(exported["meta_ch_pos_1005_distance_unit"]) == result.distance_unit
    figure = _interactive_3d_figure(result)
    trace_names = {trace.name for trace in figure.data}
    assert trace_names >= {"Scalp", "Nose (front)", "x direction", "y direction", "z direction"}
    assert "Assignment" not in trace_names
    assigned_3d = next(trace for trace in figure.data if trace.name == "Assigned slots")
    assert assigned_3d.marker.color == "#f2b51d"
    assert not trace_names.intersection({"Left ear", "Right ear"})
    axis_traces = {trace.name: trace for trace in figure.data if trace.name.endswith("direction")}
    assert axis_traces["x direction"].line.color == "#d32f2f"
    assert axis_traces["y direction"].line.color == "#2e7d32"
    assert axis_traces["z direction"].line.color == "#1565c0"
    assert figure.layout.legend.font.size == 15
    assert assigned_3d.marker.size == 8
    assert tuple(figure.layout.scene.domain.x) == (0.0, 1.0)
    assert tuple(figure.layout.scene.domain.y) == (0.0, 1.0)
    assert figure.layout.margin.l == 55
    assert figure.layout.margin.r == 55
    zoomed_figure = _interactive_3d_figure(result, camera_scale=0.8)
    assert zoomed_figure.layout.scene.camera.eye.x == 2.10 * 0.8
    assert zoomed_figure.layout.scene.camera.eye.y == 0.70 * 0.8
    assert zoomed_figure.layout.scene.camera.eye.z == 0.74 * 0.8
    preview = _alignment_2d_figure(result)
    assert {trace.name for trace in preview.data} == {
        "Head outline", "Nose (front)", "Left ear", "Right ear",
        "Standard 1005", "Aligned midpoints", "Assigned slots", "Midpoint to slot",
    }
    assert preview.layout.xaxis.scaleanchor == "y"
    assert preview.layout.xaxis.fixedrange
    assert preview.layout.yaxis.fixedrange
    assert preview.layout.dragmode is False
    assigned_trace = next(trace for trace in preview.data if trace.name == "Assigned slots")
    reference_2d = np.asarray(load_reference(result.reference)["positions_2d"])
    expected_centers = reference_2d[result.slot_ids[result.valid]]
    actual_centers = np.column_stack((assigned_trace.x, assigned_trace.y))
    np.testing.assert_array_equal(actual_centers, expected_centers)
    montage = MontageInput(
        names=("S1-D1", "S2-D1"),
        positions=np.asarray([[0.0, 0.08, 0.08], [0.03, 0.07, 0.07]]),
        optodes=OptodeGeometry(
            source_names=("S1", "S2"),
            source_positions=np.asarray([[-0.01, 0.08, 0.08], [0.02, 0.07, 0.07]]),
            detector_names=("D1",),
            detector_positions=np.asarray([[0.01, 0.08, 0.08]]),
            channel_pairs=((0, 0), (1, 0)),
        ),
    )
    optode_preview = _alignment_2d_figure(
        result,
        montage=montage,
        show_sources=True,
        show_detectors=True,
        show_sd_lines=True,
    )
    assert {trace.name for trace in optode_preview.data} == {
        "Head outline", "Nose (front)", "Left ear", "Right ear",
        "Standard 1005", "Aligned midpoints", "Assigned slots", "Midpoint to slot",
        "Sources", "Detectors", "S-D lines",
    }
    assigned_trace = next(trace for trace in optode_preview.data if trace.name == "Assigned slots")
    assert assigned_trace.marker.line.color == "#111111"
    sd_trace = next(trace for trace in optode_preview.data if trace.name == "S-D lines")
    assert sd_trace.line.width == 1
    preview_rows = _alignment_2d_rows(result)
    assert len(preview_rows) == len(result.names)
    assert set(preview_rows[0]) == {
        "channel", "midpoint_x", "midpoint_y", "assigned_slot",
        "assigned_slot_x", "assigned_slot_y", "distance", "valid"
    }
    assert figure.layout.scene.dragmode == "turntable"
    assert _plotly_review_config()["scrollZoom"] is False
    assert set(_plotly_review_config()["modeBarButtonsToRemove"]) == {
        "zoom3d", "pan3d", "orbitRotation"
    }

    reference_positions = np.asarray(load_reference()["positions"])
    scalp_vertices, scalp_triangles, _ = _standard_scalp_mesh(reference_positions)
    from scipy.spatial import ConvexHull
    envelope = ConvexHull(scalp_vertices)
    signed_distance = (
        reference_positions @ envelope.equations[:, :3].T
        + envelope.equations[:, 3]
    )
    assert scalp_triangles.shape[1] == 3
    assert np.max(signed_distance) <= 1e-10

    archive = _batch_export_zip({"sample one": result, "sample-two": result}, ["TSV", "JSON"])
    with zipfile.ZipFile(__import__("io").BytesIO(archive)) as exported:
        assert exported.namelist() == [
            "sample_one/mapping.tsv",
            "sample_one/mapping.json",
            "sample-two/mapping.tsv",
            "sample-two/mapping.json",
        ]


def test_sudan_snirf_uses_documented_axis_remap(monkeypatch, tmp_path):
    loaded = MontageInput(
        ("S1-D1",),
        np.asarray([[2.5, 3.5, 4.5]]),
        optodes=OptodeGeometry(
            ("S1",),
            np.asarray([[1.0, 2.0, 3.0]]),
            ("D1",),
            np.asarray([[4.0, 5.0, 6.0]]),
            ((0, 0),),
        ),
    )
    monkeypatch.setattr("mtrenirs.alignment.ui.load_snirf", lambda *args, **kwargs: loaded)

    montage = _load_geometry_path(
        tmp_path / "Sudan2026-cocktail-visual.snirf",
        unit="m",
        frame="head",
    )

    np.testing.assert_array_equal(montage.positions, [[-2.5, 4.5, -3.5]])
    np.testing.assert_array_equal(montage.optodes.source_positions, [[-1.0, 3.0, -2.0]])
    np.testing.assert_array_equal(montage.optodes.detector_positions, [[-4.0, 6.0, -5.0]])


def test_mtrenirs_sidecar_expands_physical_pairs_in_declared_order():
    result = align(
        positions=np.asarray([[0.0, 0.08, 0.08], [0.03, 0.07, 0.07]]),
        names=("S1-D1", "S2-D1"),
        frame="head",
    )

    with np.load(io.BytesIO(_mtrenirs_sidecar_npz(result, "pair-interleaved"))) as sidecar:
        assert sidecar["ch_names"].astype(str).tolist() == [
            "S1-D1 hbo", "S1-D1 hbr", "S2-D1 hbo", "S2-D1 hbr"
        ]
        assert sidecar["ch_types"].astype(str).tolist() == ["hbo", "hbr", "hbo", "hbr"]
        np.testing.assert_array_equal(sidecar["ch_pos_1005"][0::2], sidecar["ch_pos_1005"][1::2])
        assert str(sidecar["meta_channel_order"]) == "pair-interleaved"
        assert str(sidecar["meta_mtrenirs_role"]) == "geometry-sidecar"

    grouped_result = align(
        positions=np.vstack([result.ch_pos_1005, result.ch_pos_1005]),
        names=("S1-D1 hbo", "S2-D1 hbo", "S1-D1 hbr", "S2-D1 hbr"),
        frame="standard_1005",
    )
    with np.load(io.BytesIO(_mtrenirs_sidecar_npz(grouped_result, "pair-interleaved"))) as sidecar:
        assert sidecar["ch_names"].astype(str).tolist() == [
            "S1-D1 hbo", "S1-D1 hbr", "S2-D1 hbo", "S2-D1 hbr"
        ]


def _write_nested_bids(root):
    (root / "sub-01").mkdir(parents=True)
    (root / "sub-01" / "sub-01_optodes.tsv").write_text(
        "name\tx\ty\tz\nS1\t0\t0\t0\nD1\t2\t0\t0\n",
        encoding="utf-8",
    )
    (root / "sub-01" / "sub-01_coordsystem.json").write_text(
        json.dumps({"NIRSCoordinateUnits": "cm"}),
        encoding="utf-8",
    )
    nirs = root / "sub-01" / "ses-01" / "nirs"
    nirs.mkdir(parents=True)
    (nirs / "sub-01_ses-01_task-test_channels.tsv").write_text(
        "name\ttype\nS1_D1 760\tNIRS\nS1_D1 850\tNIRS\n",
        encoding="utf-8",
    )


def test_recursive_folder_applies_bids_inheritance(tmp_path):
    _write_nested_bids(tmp_path)

    entries = _load_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert len(entries) == 1
    assert entries[0]["label"] == "sub-01_ses-01_task-test"
    np.testing.assert_array_equal(entries[0]["montage"].positions, [[1, 0, 0]])


def test_zip_upload_preserves_nested_bids_tree(tmp_path):
    source = tmp_path / "source"
    _write_nested_bids(source)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in _discover_input_paths(source):
            zipped.write(path, path.relative_to(source))

    class Upload:
        name = "bids.zip"

        @staticmethod
        def getvalue():
            return archive.getvalue()

    entries = _load_uploaded_batch([Upload()], unit="m", frame="head")

    assert len(entries) == 1
    assert entries[0]["montage"].names == ("S1-D1",)


def test_browser_folder_upload_preserves_relative_paths(tmp_path):
    source = tmp_path / "source"
    _write_nested_bids(source)

    class Upload:
        def __init__(self, path):
            self.name = str(path.relative_to(source))
            self._data = path.read_bytes()

        def getvalue(self):
            return self._data

    uploads = [Upload(path) for path in _discover_input_paths(source)]
    entries = _load_uploaded_batch(uploads, unit="m", frame="head")

    assert len(entries) == 1
    assert entries[0]["label"] == "sub-01_ses-01_task-test"


def test_folder_scan_excludes_non_montage_files_without_failing(tmp_path):
    (tmp_path / "positions.tsv").write_text(
        "name\tx\ty\tz\nch1\t0\t1\t2\nch2\t3\t4\t5\n",
        encoding="utf-8",
    )
    (tmp_path / "events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t1\ttask\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "example", "BIDSVersion": "1.9.0"}),
        encoding="utf-8",
    )
    (tmp_path / "anatomy.txt").write_text("20 - Inferior temporal gyrus\n", encoding="utf-8")

    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert [entry["label"] for entry in entries] == ["positions"]
    assert sum(row["status"] == "Accepted" for row in report) == 1
    assert sum(row["status"] == "Skipped" for row in report) == 3
    assert not any(row["status"] == "Warning" for row in report)


def test_folder_scan_deduplicates_repeated_run_montages(tmp_path):
    content = "name\tx\ty\tz\nch1\t0\t1\t2\nch2\t3\t4\t5\n"
    (tmp_path / "run-01.tsv").write_text(content, encoding="utf-8")
    (tmp_path / "run-02.tsv").write_text(content, encoding="utf-8")

    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert len(entries) == 1
    assert len(entries[0]["source_files"]) == 2
    assert any("duplicate montage" in row["reason"] for row in report)


def test_folder_scan_groups_numbered_probe_coordinate_files(tmp_path):
    (tmp_path / "PS1.txt").write_text("CH1,-1,0,0\nCH2,-2,0,0\n", encoding="utf-8")
    (tmp_path / "PS2.txt").write_text("CH1,1,0,0\nCH2,2,0,0\n", encoding="utf-8")
    (tmp_path / "PS1_anatomy.txt").write_text("not coordinates\n", encoding="utf-8")

    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="mm", frame="head")

    assert len(entries) == 1
    assert entries[0]["montage"].names == ("PS1_CH1", "PS1_CH2", "PS2_CH1", "PS2_CH2")
    assert entries[0]["montage"].positions.shape == (4, 3)
    assert any("grouped 2" in row["reason"] for row in report)


def test_folder_scan_groups_recordings_with_identical_geometry(monkeypatch, tmp_path):
    first = tmp_path / "task-a.snirf"
    second = tmp_path / "task-b.snirf"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    montage = __import__("mtrenirs.alignment.models", fromlist=["MontageInput"]).MontageInput(
        ("ch1", "ch2"), np.asarray([[0, 0, 0], [1, 0, 0]]), "m", "head"
    )
    monkeypatch.setattr("mtrenirs.alignment.ui._load_geometry_path", lambda *args, **kwargs: montage)

    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert len(entries) == 1
    assert len(entries[0]["source_files"]) == 2
    assert any("duplicate montage" in row["reason"] for row in report)


def test_folder_scan_keeps_patient_recordings_with_distinct_geometry(monkeypatch, tmp_path):
    first = tmp_path / "sub-01.snirf"
    second = tmp_path / "sub-02.snirf"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")

    def load_by_subject(path, **_):
        offset = 0.0 if path.stem == "sub-01" else 0.01
        return __import__("mtrenirs.alignment.models", fromlist=["MontageInput"]).MontageInput(
            ("ch1", "ch2"),
            np.asarray([[offset, 0, 0], [1 + offset, 0, 0]]),
            "m",
            "head",
        )

    monkeypatch.setattr("mtrenirs.alignment.ui._load_geometry_path", load_by_subject)

    entries, _ = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert [entry["label"] for entry in entries] == ["sub-01", "sub-02"]
    assert all(len(entry["source_files"]) == 1 for entry in entries)


def test_fnirs_scan_does_not_reject_xdf_inside_eeg_folder(monkeypatch, tmp_path):
    eeg = tmp_path / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    recording = eeg / "recording.xdf"
    recording.write_bytes(b"placeholder")
    montage = __import__("mtrenirs.alignment.models", fromlist=["MontageInput"]).MontageInput(
        ("S1-D1", "S1-D2"), np.asarray([[0, 0, 0], [1, 0, 0]]), "m", "head"
    )
    monkeypatch.setattr("mtrenirs.alignment.ui._load_geometry_path", lambda *args, **kwargs: montage)

    entries, report = _scan_path_batch(
        _discover_input_paths(tmp_path), unit="m", frame="head", modality="fNIRS"
    )

    assert len(entries) == 1
    assert report[0]["status"] == "Accepted"


def test_vendor_file_is_only_warning_when_no_geometry_is_recognized(tmp_path):
    vendor = tmp_path / "device.mat"
    vendor.write_bytes(b"placeholder")

    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert not entries
    assert report[0]["status"] == "Warning"

    (tmp_path / "positions.tsv").write_text(
        "name\tx\ty\tz\nch1\t0\t1\t2\nch2\t3\t4\t5\n",
        encoding="utf-8",
    )
    entries, report = _scan_path_batch(_discover_input_paths(tmp_path), unit="m", frame="head")

    assert len(entries) == 1
    assert next(row for row in report if row["file"].endswith("device.mat"))["status"] == "Skipped"