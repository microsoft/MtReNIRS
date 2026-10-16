# MtReNIRS Models

A compact (1.22M parameters) fNIRS foundation model for extracting transferable hemodynamic
representations. MtReNIRS uses 10-second HbO/HbR windows, Standard 1005 channel
coordinates, and a spatial-temporal transformer to produce a 128-dimensional
embedding for every channel and window. This package ships the pretrained
weights plus a minimal PyTorch API.

The encoder is montage-flexible: channel positions enter through Fourier
features of canonical 3D coordinates, so recordings can use different channel
counts and layouts without retraining the encoder.

## Model contract

```text
input       normalized HbO/HbR, (B,C,T) or (C,T)
metadata    ch_types (hbo/hbr) + Standard 1005 xyz in matching channel order
fixed       10 Hz, 10-second windows, D=128
supported   Torch, Recording, NPZ; optional SNIRF, MNE Raw, and device adapters
```

> [!WARNING]
> **CHANNEL ORDER MUST MATCH**
>
> The channel axis of `signal`, `ch_types`, `ch_names`, `ch_pos_1005`, and
> `ch_pos_1005_valid` must use exactly the same order. A mismatched order can
> attach the wrong chromophore or position to a signal without changing its
> array shape.

Prefer `mtrenirs-alignment` for producing `ch_pos_1005` and its validity mask instead
of manually inventing canonical coordinates.

```text
ch_pos                 native channel midpoint coordinates in existing datasets
align(positions=...)   MtSi1005 API argument; pass ch_pos here
ch_pos_1005            aligned Standard 1005 coordinates consumed by MtReNIRS
```

In this model, a **physical S-D pair** is one source-detector geometry, while a
**measurement channel** is one signal row for that pair and chromophore. Thus,
one S-D pair with both HbO and HbR produces two model channels that share the
same position but have different `ch_types`. `C` is this measurement-channel
count: four S-D pairs with both HbO and HbR usually give `C=8`; HbO-only is
valid. The package bundles its checkpoint and requires no dataset folder
layout. It extracts embeddings only; labels and prediction heads belong to
downstream code.

## Install

```bash
git clone <mtrenirs-repository-url>
cd MtReNIRS/models && pip install -e .
```

Python 3.10 or newer is required. The same runtime dependencies are listed in
`requirements.txt`.

Installation extras select optional dependencies; they do not run processing:

| Install | Adds | Runtime entry point |
| --- | --- | --- |
| `pip install -e .` | Core PyTorch inference | `embed()`, `embed_recording()` |
| `pip install -e ".[raw]"` | MNE, SciPy, and MtSi1005 alignment | `load_snirf()`, `from_mne_raw()`, `mtrenirs.alignment.align()` |
| `pip install -e ".[muse]"` | MNE, pyxdf, and montage alignment | `load_muse_s_athena_xdf()` |

`embed_recording()` is a model method. It accepts an already canonical
`Recording`, makes 10-second windows, and runs inference. In contrast,
`mtrenirs-models[raw]` is only a pip dependency group that makes raw-data converters
available.

## Quickstart

```python
import torch
from mtrenirs.models import MtReNIRS

# Four physical S-D pairs become eight model channels: four HbO + four HbR.
# Metadata order must exactly match the signal channel axis.
ch_types = ["hbo"] * 4 + ["hbr"] * 4
ch_pos_1005 = torch.tensor([
    [-0.030, 0.080, 0.050],
    [-0.010, 0.085, 0.055],
    [ 0.030, 0.080, 0.050],
    [ 0.010, 0.085, 0.055],
    [-0.030, 0.080, 0.050],  # same four positions repeated for HbR
    [-0.010, 0.085, 0.055],
    [ 0.030, 0.080, 0.050],
    [ 0.010, 0.085, 0.055],
])

model = MtReNIRS.from_pretrained(
    ch_types=ch_types,
    ch_pos_1005=ch_pos_1005,
    device="cuda",
)

# Preprocessed fNIRS: 16 labeled trials, 8 channels, 20 seconds at 10 Hz.
# Values must be run-normalized HbO/HbR signals.
fnirs = torch.randn(16, 8, 200, device="cuda")

emb = model.embed(fnirs)  # default: (B, W, C, D) = (16, 2, 8, 128)
flat = model.embed(fnirs, pool="flatten")  # (B, W*C*D) = (16, 2048)
mean = model.embed(fnirs, pool="temporal_mean")  # (B, C, D) = (16, 8, 128)
```

## Output shapes

The encoder first produces `(B,W,C,D)`. Pooling never combines the batch axis.

| `pool` | Output | Use when |
| --- | --- | --- |
| `"none"` (default) | `(B,W,C,128)` | Whole runs, window labels, or custom pooling |
| `"flatten"` | `(B,W*C*128)` | Fixed-duration trial with one label |
| `"temporal_mean"` | `(B,C,128)` | Variable-duration trial known to contain one state |

`B` = labeled samples, `W` = 10-second windows per sample, `C` = measurement
channels. For trial labels, one trial remains one sample:

```python
labels = torch.randint(0, 3, (16,), device="cuda")
features = model.embed(fnirs, pool="flatten")  # (16, 2048)
head = torch.nn.Linear(features.shape[1], 3, device="cuda")
loss = torch.nn.functional.cross_entropy(head(features), labels)
```

For one 10-second window (`W=1`), temporal mean and flatten contain the same
values after reshaping `(B,C,128)` to `(B,C*128)`. They differ when `W>1`.

For window labels, make each window a sample (`B=number of windows`, `W=1`):

```python
windows = fnirs.unfold(-1, 100, 100)
windows = windows.permute(0, 2, 1, 3).reshape(-1, 8, 100)  # (32, 8, 100)
```

## Whole runs and trials

The model does not know task or rest boundaries. Pass a whole run with the
default `pool="none"` to keep one embedding per 10-second window:

```python
result = model.embed_recording(recording)
window_embeddings = result.embeddings  # (W,C,128)
window_onsets = result.window_onsets_seconds
```

For linear probing with one label per 10-second window, reshape each window
separately to `(W,C*128)`. Do not flatten an entire run into one sample.

Use `embed_segment()` when an event table identifies one task trial. A fixed
35-second trial can use `flatten` after applying an explicit remainder policy.
Use `temporal_mean` only when all windows in the selected segment represent the
same trial or state; averaging a run containing both rest and task is not
meaningful.

## Windowing

The default is non-overlapping 10-second windows. A final partial window is
dropped:

```text
63 s + remainder="drop"    -> 6 windows at [0, 10, 20, 30, 40, 50] s
63 s + remainder="pad"     -> 7 non-overlapping windows; final 3 s zero-padded
63 s + remainder="overlap" -> 7 windows covering both endpoints
```

```python
emb = model.embed(fnirs, pool="none")                       # default: drop
emb = model.embed(fnirs, pool="none", remainder="pad")     # keep a padded tail
emb = model.embed(fnirs, pool="none", remainder="overlap") # keep remainder
```

`"pad"` preserves non-overlap and fills the missing end of the final window
with zeros. `"overlap"` instead spreads the minimum number of full 10-second
windows across the sample. For an entire segment shorter than 10 seconds,
`embed_segment` requires `short_policy="context"` or `"pad"`.

## NPZ Input

NPZ is not a Python dictionary. It is one file containing named NumPy arrays;
`signal` is one required key. The recommended workflow is:

```text
preprocessed signal + channel metadata
                  +
montage-align -> ch_pos_1005 + valid
                  |
                  v
             Recording
                  |
                  v
            recording.npz
```

MtSi1005's `mapping.npz` contains geometry only and is therefore not a
complete MtReNIRS recording by itself. Combine it with the signal and channel
types, keeping exactly the same channel order:

```python
from mtrenirs.alignment import MtSi1005
from mtrenirs.models import Recording

mapping = MtSi1005.align(
    positions=ch_pos,                    # existing native channel midpoints
    names=channel_names,
    unit="mm",
    modality="fnirs",
)

recording = Recording(
    signal=processed_signal,             # (C,T), 10 Hz, run-normalized
    sfreq=10.0,
    ch_names=tuple(channel_names),
    ch_types=tuple(ch_types),             # hbo/hbr, length C
    ch_pos_1005=mapping.ch_pos_1005,
    ch_pos_1005_valid=mapping.valid,
    normalized=True,
)
recording.save_npz("recording.npz")
```

This is equivalent to calling `np.savez_compressed(...)` with the named arrays
below, but `Recording` validates their shapes before saving.

### Existing pretraining NPZ files

The original preprocessing output uses `X=(N,C,100)` for `N` already-windowed
10-second samples, not the canonical continuous `signal=(C,T)` key. Use those
windows directly and bind the existing canonical metadata:

```python
import numpy as np
import torch

from mtrenirs.models import MtReNIRS

with np.load(path, allow_pickle=True) as data:
    model = MtReNIRS.from_pretrained(
        ch_types=data["ch_types"].astype(str).tolist(),
        ch_pos_1005=data["ch_pos_1005"],
        ch_pos_1005_valid=data["ch_pos_1005_valid"],
    )
    embeddings = model.embed(torch.from_numpy(data["X"]), pool="none")
    # (N,1,C,128): X is already divided into 10-second samples
```

Use `Recording.from_npz()` only for the public continuous schema documented
below; it intentionally does not guess legacy `X` files.

For file-based inference:

```python
from mtrenirs.models import MtReNIRS, Recording

recording = Recording.from_npz("recording.npz")
model = MtReNIRS.from_pretrained()
result = model.embed_recording(recording)
print(result.embeddings.shape)  # (W,C,128), default pool="none"
```

Required arrays:

| Key | Shape | Description |
| --- | --- | --- |
| `signal` | `(C, T)` | Run-normalized HbO/HbR at 10 Hz |
| `sfreq` | scalar | `10.0` |
| `ch_names` | `(C,)` | Channel names |
| `ch_types` | `(C,)` | `hbo` or `hbr` |
| `ch_pos_1005` | `(C, 3)` | Standard 1005 xyz coordinates in metres |
| `normalized` | scalar | `True` |
| `ch_pos_1005_valid` | `(C,)` | Optional coordinate-validity mask |

```bash
mtrenirs-embed embed recording.npz --output embeddings.npz
mtrenirs-embed embed recording.npz --remainder pad --output embeddings-with-tail.npz
```

The output NPZ contains `embeddings`, `window_onsets_seconds`,
`window_duration_seconds`, and `pooling`.

## API

```python
model = MtReNIRS.from_pretrained(
    ch_types=ch_types,
    ch_pos_1005=ch_pos_1005,
    device="cpu",
)
```

Loads `weights/mtrenirs_10s.pt` and binds the encoder to the channel metadata.
Provide an explicit checkpoint path only when intentionally loading different
weights.

```python
embeddings = model.embed(fnirs)
```

`fnirs` is a normalized Torch tensor shaped `(B,C,T)` or `(C,T)`. Pooling is
`"none"` (`B,W,C,128`), `"flatten"` (`B,W*C*128`), or `"temporal_mean"`
(`B,C,128`). The default is `"none"`; channel pooling is not exposed.

```python
result = model.embed_segment(recording, onset=30, duration=35, remainder="pad")
```

Processes an optional portion of a file. `duration` selects part of the
recording; the encoder window remains fixed at 10 seconds. Set
`remainder="pad"` to retain a final partial window without overlap.

## fNIRS Preprocessing

The model expects HbO/HbR concentration signals processed as follows:

1. Convert intensity to optical density.
2. Apply TDDR motion correction.
3. Convert to HbO/HbR with MBLL.
4. Band-pass filter from 0.01 to 0.5 Hz.
5. Resample to 10 Hz.
6. Normalize each channel over the run and clip to `[-6, 6]`.
7. Assign each channel to Standard 1005 coordinates with MtSi1005.

Install `mtrenirs-models[raw]` to use `mtrenirs.models.preprocessing.load_snirf()` or
`from_mne_raw()`. Install `mtrenirs-models[muse]` only when the explicit Muse S Athena
adapter is needed.

## Using MtSi1005

Install the sibling package alongside MtReNIRS when Standard 1005 coordinates are
not already present:

```bash
pip install -e ../alignment
pip install -e ".[raw]"
```

The MNE/SNIRF preprocessing path can align and embed a recording directly:

```python
from mtrenirs.models import MtReNIRS
from mtrenirs.models.preprocessing import load_snirf

recording = load_snirf("recording.snirf", align_positions=True)
model = MtReNIRS.from_pretrained()
result = model.embed_recording(recording)
```

An existing MNE Raw object uses the same path:

```python
from mtrenirs.models.preprocessing import from_mne_raw

recording = from_mne_raw(raw, align_positions=True)
result = MtReNIRS.from_pretrained().embed_recording(recording, pool="none")
```

Muse S Athena XDF uses the same signal-processing path after its
`Muse-OPTICS` PPG stream is converted to an MNE continuous-wave fNIRS Raw:

```python
from mtrenirs.models import MtReNIRS
from mtrenirs.models.adapters.muse_s_athena import load_muse_s_athena_xdf

recording = load_muse_s_athena_xdf(
    "sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
    align_positions=True,
)

recording.save_npz("sub-P001_run-001_recording.npz") # optional; save preprocessed file 

result = MtReNIRS.from_pretrained().embed_recording(recording)
```

The adapter selects the eight Muse intensity columns, constructs four
source-detector pairs at 730/850 nm, and then applies OD, TDDR, MBLL, filtering,
10 Hz resampling, run normalization, and the fixed Muse S Athena Standard 1005
alignment. The BIDS folder layout organizes the recording but does not itself
turn XDF into MNE Raw; `load_muse_s_athena_xdf()` performs that conversion and
does not depend on a particular folder name. Other XDF devices are rejected
rather than assigned the Muse layout.

For SNIRF/MNE fNIRS, `raw.info["chs"][i]["loc"][:3]` is the channel midpoint
in metres. Two wavelengths or HbO/HbR from one S-D pair remain separate
measurement channels that share the same midpoint.

For already preprocessed tensors, use MtSi1005 explicitly and pass its
result to the model. `channel_xyz`, `channel_names`, `ch_types`, and the signal
must all use the same channel order:

```python
from mtrenirs.alignment import MtSi1005

mapping = MtSi1005.align(
    positions=ch_pos,
    names=channel_names,
    unit="mm",
    modality="fnirs",
)
model = MtReNIRS.from_pretrained(
    ch_types=ch_types,
    ch_pos_1005=mapping.ch_pos_1005,
    ch_pos_1005_valid=mapping.valid,
)
embeddings = model.embed(fnirs)
```

Inspect `mapping.distances` and `mapping.collisions` as alignment QC before
inference. The packages exchange only canonical coordinates and validity flags;
MtReNIRS does not require the original montage table.

## Structure

```text
mtrenirs/models/
    model.py          MtReNIRS: load / embed / file inference
    network.py        Fourier embeddings + transformer encoder
    recording.py      canonical NPZ recording format
    segmentation.py   10-second windowing and remainder policy
    preprocessing.py  optional MNE/SNIRF preprocessing
    cli.py            `mtrenirs-embed embed`
examples/
    extract_embeddings.py
    MtReNIRS_quickstart.ipynb  pooling, windowing, preprocessing, and linear probes
weights/
    mtrenirs_10s.pt   pretrained encoder weights
```

Open the notebook with the project's Python kernel in VS Code.

## Privacy

This package does not include, collect, transmit, or store third-party fNIRS
recordings. Inference runs locally. The repository contains only model weights,
code, generated tests, and approved rendered montage images in the separate
MtSi1005 package.

## Scope

The public package provides pretrained embeddings. Probe training, LoRA,
pretraining, and full fine-tuning remain research workflows outside this
inference API.
