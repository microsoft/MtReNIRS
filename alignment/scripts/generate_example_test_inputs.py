# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mtrenirs.alignment import align, load_reference  # noqa: E402
from mtrenirs.alignment.io import (  # noqa: E402
    load_bids_nirs_sidecars,
    load_json,
    load_npz,
    load_position_text,
    load_positions_tsv,
    load_snirf,
)


OUTPUT_ROOT = REPOSITORY_ROOT / "example_test_input"
SYNTHETIC_FNIRS_SOURCES_MM = np.asarray(
    [
        [-60.0, 45.0, 55.0],
        [-30.0, 75.0, 60.0],
        [0.0, 85.0, 65.0],
        [30.0, 75.0, 60.0],
        [60.0, 45.0, 55.0],
        [-50.0, -20.0, 55.0],
        [0.0, -55.0, 60.0],
        [50.0, -20.0, 55.0],
    ]
)
SYNTHETIC_FNIRS_DETECTORS_MM = np.asarray(
    [
        [-35.0, 45.0, 55.0],
        [-5.0, 75.0, 60.0],
        [20.0, 85.0, 65.0],
        [55.0, 75.0, 60.0],
        [35.0, 45.0, 55.0],
        [-25.0, -20.0, 55.0],
        [20.0, -55.0, 60.0],
        [25.0, -20.0, 55.0],
    ]
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_expected(path: Path, result) -> None:
    _write_json(
        path,
        {
            "method": result.method,
            "rows": len(result.names),
            "valid_rows": int(result.valid.sum()),
            "slot_labels": list(result.slot_labels),
            "distance_unit": result.distance_unit,
            "collision_slots": {
                result.slot_labels[next(index for index, value in enumerate(result.slot_ids) if value == slot)]: list(names)
                for slot, names in result.collisions.items()
            },
        },
    )


def _write_snirf(path: Path) -> None:
    sources = SYNTHETIC_FNIRS_SOURCES_MM / 1000.0
    detectors = SYNTHETIC_FNIRS_DETECTORS_MM / 1000.0
    pairs = tuple((index, index) for index in range(1, len(sources) + 1))
    measurements = [
        (source, detector, wavelength)
        for source, detector in pairs
        for wavelength in (1, 2)
    ]

    with h5py.File(path, "w") as file:
        file.create_dataset("formatVersion", data=np.bytes_("1.1"))
        nirs = file.create_group("nirs")
        metadata = nirs.create_group("metaDataTags")
        for key, value in {
            "SubjectID": "synthetic-01",
            "MeasurementDate": "2000-01-01",
            "MeasurementTime": "00:00:00Z",
            "LengthUnit": "m",
            "TimeUnit": "s",
        }.items():
            metadata.create_dataset(key, data=np.bytes_(value))

        probe = nirs.create_group("probe")
        probe.create_dataset("wavelengths", data=np.asarray([760.0, 850.0]))
        probe.create_dataset("sourcePos3D", data=sources)
        probe.create_dataset("detectorPos3D", data=detectors)

        data = nirs.create_group("data1")
        data.create_dataset("dataTimeSeries", data=np.zeros((10, len(measurements))))
        data.create_dataset("time", data=np.asarray([0.0, 0.1]))
        for index, (source, detector, wavelength) in enumerate(measurements, 1):
            measurement = data.create_group(f"measurementList{index}")
            measurement.create_dataset("sourceIndex", data=np.asarray([source]))
            measurement.create_dataset("detectorIndex", data=np.asarray([detector]))
            measurement.create_dataset("wavelengthIndex", data=np.asarray([wavelength]))
            measurement.create_dataset("dataType", data=np.asarray([1]))


def _write_readme() -> None:
    (OUTPUT_ROOT / "README.md").write_text(
        """# Synthetic example test inputs

Every coordinate and channel name in this directory was authored for this
package. Nothing was copied, perturbed, averaged, or transformed from a patient
or pretraining dataset. The examples are deterministic and designed to be
distributed with the package without third-party dataset material.

Each scenario includes `expected_alignment.json`. It records the method, valid
row count, assigned Standard 1005 labels, distance unit, and collisions expected
from the current package. Use it to understand the result before running the UI.

| Scenario | Input format | Rows/pairs | What it demonstrates | Expected behavior |
|---|---|---:|---|---|
| `01-canonical-labels` | JSON | 10 | Known Standard 1005 labels with zero coordinate placeholders | Exact label assignment; coordinates are ignored |
| `02-positions-3d` | TSV | 10 | Generic 3D channel positions in metres | Isotropic 3D alignment and nearest-slot assignment |
| `03-positions-2d` | TXT | 9 | Header-based planar coordinates in metres | Isotropic 2D projection alignment |
| `04-standard-frame` | NPZ | 10 | Coordinates already in `standard_1005` | Canonical-frame 3D nearest assignment |
| `05-missing-position` | JSON | 9 | One `null` coordinate row | Eight finite rows map; missing row remains invalid |
| `06-bids-fnirs` | BIDS TSV/JSON | 8 pairs / 16 NIRS rows | Sources, detectors, duplicate wavelengths, and an ignored non-NIRS row | One midpoint per physical S-D pair |
| `07-minimal-snirf` | SNIRF | 8 pairs / 16 measurement rows | A valid minimal continuous-wave container | UI deduplicates physical pairs; zero-valued signal is only a placeholder |
| `08-canonical-collision` | JSON | 10 | Multiple rows targeting repeated canonical slots | Valid mappings plus two reported collisions |

## Units and UI settings

The standalone TSV and TXT examples use metres, matching the UI default. JSON,
NPZ, BIDS, and SNIRF carry their own unit metadata. BIDS and SNIRF S-D distances
are between 8 and 35 mm, so they remain visible under the UI's physical-pair
filter. The UI produces 8 geometry rows for each fNIRS scenario. The SNIRF CLI
preserves its 16 wavelength measurement rows because single-file CLI loading
does not enable physical-pair deduplication.

## Why XDF is not included

The package can read XDF files containing per-channel `x`, `y`, and `z`
metadata, but the supported dependency (`pyxdf`) is a reader and does not ship a
writer. This repository does not include a hand-crafted or renamed fake binary.
An XDF example should be added only when generated and validated by an official
XDF writer.

## Regeneration

From the repository root:

```bash
python scripts/generate_example_test_inputs.py
pytest -q tests/test_example_inputs.py
```

The generator replaces this directory completely, which prevents old private
fixtures or symlinks from surviving regeneration.
""",
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir()

    scenario = OUTPUT_ROOT / "01-canonical-labels"
    scenario.mkdir()
    canonical_path = scenario / "canonical_labels.json"
    canonical_labels = ("Fp1", "Fp2", "F7", "F8", "C3", "Cz", "C4", "P3", "P4", "Oz")
    _write_json(
        canonical_path,
        {
            "ch_names": [
                "left-front",
                "right-front",
                "left-frontal",
                "right-frontal",
                "left-center",
                "center",
                "right-center",
                "left-back",
                "right-back",
                "back-center",
            ],
            "ch_pos": [[0.0, 0.0, 0.0]] * len(canonical_labels),
            "canonical_labels": list(canonical_labels),
            "meta_ch_pos_unit": "m",
            "meta_ch_pos_frame": "head",
        },
    )
    _write_expected(scenario / "expected_alignment.json", align(load_json(canonical_path), canonical_labels=canonical_labels))

    scenario = OUTPUT_ROOT / "02-positions-3d"
    scenario.mkdir()
    tsv_path = scenario / "positions_3d.tsv"
    tsv_path.write_text(
        "name\tx\ty\tz\n"
        "front-center\t0.000\t0.085\t0.060\n"
        "front-left\t-0.055\t0.070\t0.055\n"
        "front-right\t0.055\t0.070\t0.055\n"
        "left-upper\t-0.075\t0.025\t0.060\n"
        "left-lower\t-0.075\t-0.025\t0.055\n"
        "center\t0.000\t0.000\t0.085\n"
        "right-upper\t0.075\t0.025\t0.060\n"
        "right-lower\t0.075\t-0.025\t0.055\n"
        "back-left\t-0.045\t-0.065\t0.050\n"
        "back-right\t0.045\t-0.065\t0.050\n",
        encoding="utf-8",
    )
    _write_expected(scenario / "expected_alignment.json", align(load_positions_tsv(tsv_path)))

    scenario = OUTPUT_ROOT / "03-positions-2d"
    scenario.mkdir()
    txt_path = scenario / "positions_2d.txt"
    txt_path.write_text(
        "channel x y\n"
        "front 0.000 0.080\n"
        "front-left -0.050 0.055\n"
        "front-right 0.050 0.055\n"
        "left -0.070 0.000\n"
        "center 0.000 0.000\n"
        "right 0.070 0.000\n"
        "back-left -0.045 -0.055\n"
        "back 0.000 -0.070\n"
        "back-right 0.045 -0.055\n",
        encoding="utf-8",
    )
    _write_expected(scenario / "expected_alignment.json", align(load_position_text(txt_path)))

    scenario = OUTPUT_ROOT / "04-standard-frame"
    scenario.mkdir()
    npz_path = scenario / "standard_frame.npz"
    reference = load_reference("standard_1005")
    reference_lookup = dict(zip(reference["labels"], reference["positions"]))
    standard_labels = ("Fp1", "Fp2", "F7", "F8", "C3", "Cz", "C4", "P3", "P4", "Oz")
    np.savez_compressed(
        npz_path,
        ch_names=np.asarray(standard_labels),
        ch_pos=np.asarray([reference_lookup[label] for label in standard_labels]),
        meta_ch_pos_unit=np.asarray("m"),
        meta_ch_pos_frame=np.asarray("standard_1005"),
    )
    _write_expected(scenario / "expected_alignment.json", align(load_npz(npz_path)))

    scenario = OUTPUT_ROOT / "05-missing-position"
    scenario.mkdir()
    missing_path = scenario / "missing_position.json"
    _write_json(
        missing_path,
        {
            "ch_names": [
                "front",
                "front-left",
                "missing",
                "front-right",
                "left",
                "center",
                "right",
                "back-left",
                "back-right",
            ],
            "ch_pos": [
                [0.000, 0.080, 0.060],
                [-0.050, 0.060, 0.055],
                [None, None, None],
                [0.050, 0.060, 0.055],
                [-0.070, 0.000, 0.060],
                [0.000, 0.000, 0.085],
                [0.070, 0.000, 0.060],
                [-0.045, -0.060, 0.050],
                [0.045, -0.060, 0.050],
            ],
            "meta_ch_pos_unit": "m",
            "meta_ch_pos_frame": "head",
        },
    )
    _write_expected(scenario / "expected_alignment.json", align(load_json(missing_path)))

    scenario = OUTPUT_ROOT / "06-bids-fnirs"
    scenario.mkdir()
    optodes_path = scenario / "sub-synthetic_optodes.tsv"
    optode_rows = ["name\ttype\tx\ty\tz"]
    for index, position in enumerate(SYNTHETIC_FNIRS_SOURCES_MM, 1):
        optode_rows.append(f"S{index}\tsource\t{position[0]:g}\t{position[1]:g}\t{position[2]:g}")
    for index, position in enumerate(SYNTHETIC_FNIRS_DETECTORS_MM, 1):
        optode_rows.append(f"D{index}\tdetector\t{position[0]:g}\t{position[1]:g}\t{position[2]:g}")
    optodes_path.write_text("\n".join(optode_rows) + "\n", encoding="utf-8")
    channels_path = scenario / "sub-synthetic_channels.tsv"
    channel_rows = ["name\ttype\tsource\tdetector\twavelength_nominal"]
    for index in range(1, len(SYNTHETIC_FNIRS_SOURCES_MM) + 1):
        for wavelength in (760, 850):
            channel_rows.append(
                f"S{index}_D{index} {wavelength}\tNIRS_CW_AMPLITUDE\t"
                f"S{index}\tD{index}\t{wavelength}"
            )
    channel_rows.append("AUX1\tAUX\tn/a\tn/a\tn/a")
    channels_path.write_text("\n".join(channel_rows) + "\n", encoding="utf-8")
    coordsystem_path = scenario / "sub-synthetic_coordsystem.json"
    _write_json(
        coordsystem_path,
        {
            "NIRSCoordinateSystem": "Other",
            "NIRSCoordinateUnits": "mm",
            "NIRSCoordinateSystemDescription": "Package-authored synthetic head coordinates",
        },
    )
    bids_montage = load_bids_nirs_sidecars(
        optodes_path,
        coordsystem_path,
        channels_path,
        min_sd_distance_mm=8.0,
        max_sd_distance_mm=35.0,
    )
    _write_expected(scenario / "expected_alignment.json", align(bids_montage))

    scenario = OUTPUT_ROOT / "07-minimal-snirf"
    scenario.mkdir()
    snirf_path = scenario / "synthetic_recording.snirf"
    _write_snirf(snirf_path)
    snirf_montage = load_snirf(
        snirf_path,
        physical_pairs=True,
        min_sd_distance_mm=8.0,
        max_sd_distance_mm=35.0,
    )
    _write_expected(scenario / "expected_alignment.json", align(snirf_montage))

    scenario = OUTPUT_ROOT / "08-canonical-collision"
    scenario.mkdir()
    collision_path = scenario / "canonical_collision.json"
    collision_labels = ("Fp1", "Fp1", "Fp2", "F7", "F8", "C3", "Cz", "C4", "Oz", "Oz")
    _write_json(
        collision_path,
        {
            "ch_names": [
                "left-a",
                "left-b",
                "right-front",
                "left-frontal",
                "right-frontal",
                "left-center",
                "center",
                "right-center",
                "back-a",
                "back-b",
            ],
            "ch_pos": [[0.0, 0.0, 0.0]] * len(collision_labels),
            "canonical_labels": list(collision_labels),
            "meta_ch_pos_unit": "m",
            "meta_ch_pos_frame": "head",
        },
    )
    _write_expected(scenario / "expected_alignment.json", align(load_json(collision_path), canonical_labels=collision_labels))

    _write_readme()
    print(f"Generated 8 synthetic scenarios in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()