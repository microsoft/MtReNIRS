# Synthetic example test inputs

Every coordinate and channel name in this directory was authored for this
package. Nothing was copied, perturbed, averaged, or transformed from a patient
or pretraining dataset. The examples are deterministic and designed to be
distributed with the package without third-party dataset material.

Each scenario includes `expected_alignment.json`. It records the method, valid
row count, assigned Standard 1005 labels, distance unit, and collisions expected
from the current package. Use it to understand the result before running the UI.

| Scenario | Input format | Input size | What it demonstrates | Expected behavior |
|---|---|---:|---|---|
| `01-canonical-labels` | JSON | 10 channels | Known Standard 1005 labels with zero coordinate placeholders | Exact label assignment; coordinates are ignored |
| `02-positions-3d` | TSV | 10 channels | Generic 3D channel positions in metres | Isotropic 3D alignment and nearest-slot assignment |
| `03-positions-2d` | TXT | 9 channels | Header-based planar coordinates in metres | Isotropic 2D projection alignment |
| `04-standard-frame` | NPZ | 10 channels | Coordinates already in `standard_1005` | Canonical-frame 3D nearest assignment |
| `05-missing-position` | JSON | 9 channels | One `null` coordinate row | Eight finite rows map; missing row remains invalid |
| `06-bids-fnirs` | BIDS TSV/JSON | 8 physical pairs / 16 measurements | Sources, detectors, duplicate wavelengths, and an ignored non-NIRS row | One midpoint per physical S-D pair |
| `07-minimal-snirf` | SNIRF | 8 physical pairs / 16 measurements | A valid minimal continuous-wave container | UI deduplicates physical pairs; zero-valued signal is only a placeholder |
| `08-canonical-collision` | JSON | 10 channels | Multiple rows targeting repeated canonical slots | Valid mappings plus two reported collisions |

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
