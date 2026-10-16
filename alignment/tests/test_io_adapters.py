# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys
from types import SimpleNamespace

import numpy as np

from mtrenirs.alignment.alignment import align
from mtrenirs.alignment.io import load_bids_nirs_sidecars, load_position_text, load_xdf


def test_load_position_text_accepts_whitespace_columns(tmp_path):
    path = tmp_path / "positions.txt"
    path.write_text("channel x y z\nch1 0 1 2\nch2 3 4 5\n", encoding="utf-8")

    montage = load_position_text(path, unit="mm")

    assert montage.names == ("ch1", "ch2")
    np.testing.assert_array_equal(montage.positions, [[0, 1, 2], [3, 4, 5]])
    assert montage.unit == "mm"


def test_load_position_text_accepts_headerless_coordinate_rows(tmp_path):
    path = tmp_path / "probe.txt"
    path.write_text("CH1,-71,-33,-19\nCH2,-72,-5,-18\n", encoding="utf-8")

    montage = load_position_text(path, unit="mm")

    assert montage.names == ("CH1", "CH2")
    np.testing.assert_array_equal(montage.positions, [[-71, -33, -19], [-72, -5, -18]])


def test_load_position_text_recovers_missing_first_separator(tmp_path):
    path = tmp_path / "probe.txt"
    path.write_text("CH1,-71,-33,-19\nCH2-72,-5,-18\n", encoding="utf-8")

    montage = load_position_text(path, unit="mm")

    assert montage.names == ("CH1", "CH2")
    np.testing.assert_array_equal(montage.positions, [[-71, -33, -19], [-72, -5, -18]])


def test_load_bids_nirs_sidecars_builds_channel_midpoints(tmp_path):
    optodes = tmp_path / "sub-01_optodes.tsv"
    optodes.write_text("name\tx\ty\tz\nS1\t0\t0\t0\nD1\t2\t0\t0\n", encoding="utf-8")
    channels = tmp_path / "sub-01_channels.tsv"
    channels.write_text("name\ttype\nS1_D1 760\tNIRS\nS1_D1 850\tNIRS\n", encoding="utf-8")
    coordsystem = tmp_path / "sub-01_coordsystem.json"
    coordsystem.write_text(json.dumps({"NIRSCoordinateUnits": "cm"}), encoding="utf-8")

    montage = load_bids_nirs_sidecars(optodes, coordsystem, channels)

    assert montage.names == ("S1-D1",)
    np.testing.assert_array_equal(montage.positions, [[1, 0, 0]])
    assert montage.unit == "cm"


def test_load_xdf_recognizes_muse_s_athena_optics(monkeypatch, tmp_path):
    streams = [
        {
            "info": {
                "name": ["Muse-OPTICS (00:00:00:00:00:00)"],
                "type": ["PPG"],
                "channel_count": ["16"],
            }
        }
    ]
    monkeypatch.setitem(
        sys.modules,
        "pyxdf",
        SimpleNamespace(load_xdf=lambda *args, **kwargs: (streams, {})),
    )

    montage = load_xdf(tmp_path / "recording.xdf")

    assert montage.names == ("S1-D1", "S1-D2", "S1-D3", "S1-D4")
    assert montage.positions.shape == (4, 3)
    assert montage.unit == "m"
    assert montage.frame == "standard_1005"
    assert montage.modality == "fnirs"
    assert align(montage).slot_labels == ("Fp1h", "Fp2h", "Fpz", "Fpz")