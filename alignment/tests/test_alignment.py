import numpy as np

from mtrenirs.alignment import align, align_optodes, from_mne_raw, load_reference, load_snirf
from mtrenirs.alignment.cli import main
from mtrenirs.alignment.io import load_json, load_npz


def test_canonical_labels_map_exactly() -> None:
    result = align(
        positions=np.zeros((3, 3)),
        names=["a", "b", "c"],
        canonical_labels=["Fp1", "Cz", "O2"],
    )
    assert result.slot_labels == ("Fp1", "Cz", "O2")
    assert result.valid.all()
    assert np.isnan(result.source_positions).all()
    assert np.isnan(result.distances).all()
    assert result.distance_unit == "n/a"


def test_reference_points_return_to_their_slots() -> None:
    reference = load_reference()
    ids = [0, 50, 100, 200]
    points = np.asarray(reference["positions"])[ids]
    result = align(
        positions=points,
        names=[f"ch{i}" for i in ids],
        frame="standard_1005",
    )
    assert result.valid.all()
    assert result.slot_ids.tolist() == ids


def test_optode_pairs_create_channels() -> None:
    reference = load_reference()
    points = np.asarray(reference["positions"])
    result = align_optodes(
        source_positions=points[[0, 1]],
        detector_positions=points[[2, 3]],
        channel_pairs=[(0, 0), (1, 1)],
        frame="standard_1005",
    )
    assert result.names == ("S1_D1", "S2_D2")
    assert result.ch_pos_1005.shape == (2, 3)
    assert result.valid.all()


def test_collisions_use_physical_source_detector_pairs() -> None:
    result = align(
        positions=np.zeros((3, 3)),
        names=["S1_D1 hbo", "S1_D1 hbr", "S2_D1 760"],
        canonical_labels=["Fp1", "Fp1", "Fp1"],
    )
    slot_id = int(result.slot_ids[0])
    assert result.collisions == {slot_id: ("S1_D1", "S2_D1")}


def test_saved_json_and_npz_are_reusable_inputs(tmp_path) -> None:
    result = align(
        positions=np.zeros((3, 3)),
        names=["a", "b", "c"],
        canonical_labels=["Fp1", "Cz", "O2"],
    )
    json_path = result.save_json(tmp_path / "mapping.json")
    npz_path = result.save_npz(tmp_path / "mapping.npz")

    for montage in (load_json(json_path), load_npz(npz_path)):
        restored = align(montage)
        assert restored.slot_labels == result.slot_labels
        assert np.allclose(restored.ch_pos_1005, result.ch_pos_1005)


def test_cli_writes_all_mapping_formats(tmp_path) -> None:
    reference = load_reference()
    input_path = tmp_path / "positions.npz"
    np.savez_compressed(
        input_path,
        ch_names=np.asarray(["a", "b", "c"]),
        ch_pos=np.asarray(reference["positions"])[[0, 50, 100]],
        meta_ch_pos_frame=np.asarray("standard_1005"),
    )
    output = tmp_path / "output"

    assert main([str(input_path), "--output-dir", str(output)]) == 0
    assert {path.name for path in output.iterdir()} == {
        "mapping.tsv",
        "mapping.json",
        "mapping.npz",
        "montage_before_after.png",
    }


def test_snirf_loader_uses_mne_channel_locations(monkeypatch, tmp_path) -> None:
    class FakeRaw:
        ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760"]
        info = {
            "chs": [
                {"loc": np.asarray([-0.03, 0.08, 0.05] + [0.0] * 9)},
                {"loc": np.asarray([-0.03, 0.08, 0.05] + [0.0] * 9)},
                {"loc": np.asarray([0.03, 0.08, 0.05] + [0.0] * 9)},
            ]
        }

    monkeypatch.setattr("mne.io.read_raw_snirf", lambda *args, **kwargs: FakeRaw())
    direct = from_mne_raw(FakeRaw())
    montage = load_snirf(tmp_path / "recording.snirf")

    assert montage.names == tuple(FakeRaw.ch_names)
    assert montage.positions.shape == (3, 3)
    assert montage.unit == "m"
    assert montage.frame == "head"
    assert np.array_equal(montage.positions, direct.positions)
