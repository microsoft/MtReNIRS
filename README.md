<div align="center">
    <img src="./assets/mtrenirs-scheme.jpeg" alt="MtReNIRS" width="720">
</div>

<div align="center">


## MtReNIRS: Montage-aligned Transferable Representations for fNIRS

[Models](models/) · [Montage alignment](alignment/) · **Paper coming soon**

</div>

---

MtReNIRS is our most capable fNIRS foundation model to date. It learns
transferable neurovascular representations from large-scale, heterogeneous
functional near-infrared spectroscopy (fNIRS) data. The current model encodes
10-second HbO/HbR windows for downstream neuroimaging and brain-computer
interface research across different channel counts and sensor layouts.

The project also includes **MtSi1005**, an independently installable montage
alignment toolkit that maps heterogeneous sensor layouts into canonical 10-05
space for MtReNIRS inference or standalone geometry workflows.

## Key features

- **Transferable fNIRS representations:** a pretrained spatial-temporal encoder
    for linear probing, few-shot learning, LoRA, and full fine-tuning.
- **Montage-flexible inference:** channel coordinates are encoded in canonical
    10-05 space, allowing variable channel counts and layouts.
- **Low-label adaptation:** the frozen 10-second model provides competitive
    cross-subject transfer and strong within-subject few-shot performance.
- **Modular ecosystem:** install the model and alignment toolkit together or
    use MtSi1005 independently without PyTorch.

## Project structure

```text
MtReNIRS/
├── models/       # Model package, checkpoint, README, tests, and examples
├── alignment/    # Independent MtSi1005 package, README, tests, and examples
└── README.md     # Project overview
```

`models/` publishes `mtrenirs-models` as `mtrenirs.models`; `alignment/`
publishes `mtrenirs-alignment` as `mtrenirs.alignment`. MtSi1005 does not depend
on MtReNIRS or PyTorch, and standardized model inputs do not require the
alignment package.

## Installation

Install both packages for end-to-end development:

```bash

git clone https://github.com/msr-central/MtReNIRS

cd MtReNIRS

pip install -e ./alignment -e ./models
```

Or install either component independently:

```bash
pip install -e ./models
pip install -e ./alignment
```

Optional raw SNIRF/MNE and Muse adapters are available through model extras:

```bash
pip install "mtrenirs-models[raw]"
pip install "mtrenirs-models[muse]"
```

## Quickstart

```python
from mtrenirs.alignment import MtSi1005
from mtrenirs.models import MtReNIRS

mapping = MtSi1005.align(
    positions=channel_positions,
    names=channel_names,
    unit="mm",
)

model = MtReNIRS.from_pretrained(
    "mtrenirs-10s",
    ch_types=channel_types,
    ch_pos_1005=mapping.ch_pos_1005,
    ch_pos_1005_valid=mapping.valid,
)

embeddings = model.embed(signal)
```

For alignment-only use:

```python
from mtrenirs.alignment import MtSi1005

mapping = MtSi1005.align(
    positions=channel_positions,
    names=channel_names,
    unit="mm",
)
mapping.save_npz("mapping.npz")
```

The functional alignment API remains available as
`from mtrenirs.alignment import align, load_reference`.

## Command-line tools

```bash
mtrenirs-embed embed recording.npz --output embeddings.npz
mtsi1005 positions.tsv --unit mm --output-dir output
mtsi1005-ui
```

The established `montage-align` and `montage-align-ui` command aliases are also
retained for alignment-tool users.

## Results snapshot

The following results use the current **10-second MtReNIRS configuration**.
Full protocols, subject-level statistics, ablations, and reproducibility
details will accompany the forthcoming paper.

### Cross-subject transfer (LOSO)

Mean ± standard deviation of macro-F1 across held-out subjects. The best
MtReNIRS adaptation result for each dataset is bolded.

| Adaptation | Gunther 2014 (`n=8`) | Shader 2021 (`n=8`) | SujinBak 2019 (`n=30`) |
| --- | ---: | ---: | ---: |
| Linear probe | **0.873 ± 0.078** | 0.445 ± 0.094 | 0.679 ± 0.177 |
| MLP probe | 0.865 ± 0.077 | 0.447 ± 0.103 | 0.672 ± 0.186 |
| LoRA | 0.862 ± 0.052 | 0.437 ± 0.116 | **0.703 ± 0.186** |
| Full fine-tuning | 0.838 ± 0.078 | **0.473 ± 0.056** | 0.702 ± 0.191 |
| Scratch fine-tuning | 0.823 ± 0.077 | 0.402 ± 0.062 | 0.667 ± 0.162 |

### Within-subject few-shot learning

Frozen linear-probe macro-F1. Values are mean ± population standard deviation
across subjects after averaging 50 seeds within each subject.

| Dataset | 2-shot | 4-shot | 6-shot | 8-shot |
| --- | ---: | ---: | ---: | ---: |
| Gunther 2014 | 0.778 ± 0.116 | 0.844 ± 0.095 | 0.875 ± 0.080 | **0.896 ± 0.070** |
| Shader 2021 | 0.391 ± 0.070 | 0.430 ± 0.097 | **0.442 ± 0.119** | 0.431 ± 0.121 |
| SujinBak 2019 | 0.404 ± 0.081 | 0.462 ± 0.119 | 0.500 ± 0.142 | **0.530 ± 0.151** |

### Gunther 2014 comparison with prior work (fNIRS-only pretraining)

Within-subject few-shot **accuracy** for mental arithmetic versus rest. Prior
results are reproduced from the cited source used in the forthcoming
manuscript. Differences in preprocessing, sampling, patch duration, and
evaluation make this a contextual comparison rather than a controlled
head-to-head benchmark.

| Pretrained model | 2-shot | 4-shot | 6-shot | 8-shot |
| --- | ---: | ---: | ---: | ---: |
| **MtReNIRS (ours, 10 s)** | **0.78 ± 0.11** | **0.85 ± 0.09** | **0.88 ± 0.08** | **0.90 ± 0.07** |
| [MAE](https://doi.org/10.1109/BCI65088.2025.10931275) | 0.67 ± 0.26 | 0.67 ± 0.12 | 0.75 ± 0.10 | 0.75 ± 0.10 |
| [SSCL](https://doi.org/10.1109/IJCNN52387.2021.9533305) | 0.70 ± 0.12 | 0.73 ± 0.13 | 0.69 ± 0.22 | 0.69 ± 0.22 |
| [SleepFM](https://doi.org/10.48550/arXiv.2405.17766) | 0.75 ± 0.06 | 0.62 ± 0.09 | 0.60 ± 0.07 | 0.58 ± 0.13 |
| [EFRM](https://doi.org/10.1016/j.compbiomed.2025.111292) | 0.67 ± 0.26 | 0.67 ± 0.12 | 0.75 ± 0.10 | 0.75 ± 0.10 |

### Downstream Test datasets

The following datasets are used only for downstream evaluation and are not included in [pretraining dataset sources](https://github.com/microsoft/MtReNIRS-Montage-aligned-Transferable-Representations-for-fNIRS-/tree/main/alignment#original-pretraining-dataset-sources).

| Dataset | Original data source |
| --- | --- |
| Gunther2014 | G. Bauernfeind, R. Scherer, G. Pfurtscheller, and C. Neuper, “Single-trial classification of antagonistic oxyhemoglobin responses during mental arithmetic,” *Medical & Biological Engineering & Computing*, vol. 49, no. 9, pp. 979–984, 2011. [Online]. Available: https://doi.org/10.1007/s11517-011-0792-5 |
| Shader2021 | M. J. Shader, R. Luke, N. Gouailhardou, and C. M. McKay, “The use of broad vs restricted regions of interest in functional near-infrared spectroscopy for measuring cortical activation to auditory-only and visual-only speech,” *Hearing Research*, vol. 406, p. 108256, 2021. [Online]. Available: https://doi.org/10.1016/j.heares.2021.108256 |
| SujinBak2019 | S. Bak, J. Park, J. Shin, and J. Jeong, “Open-access fNIRS dataset for classification of unilateral finger- and foot-tapping,” *Electronics*, vol. 8, no. 12, p. 1486, 2019. [Online]. Available: https://doi.org/10.3390/electronics8121486 |



## Documentation

- [MtReNIRS model guide](models/README.md)
- [MtSi1005 alignment tookit guide](alignment/README.md)
- [Model quickstart notebook](models/examples/MtReNIRS_quickstart.ipynb)
- [Montage alignment examples](alignment/examples_opensource_fnirs/pretraining_montage_examples.ipynb)

## Project policies

- Contributions: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security reporting: [SECURITY.md](SECURITY.md)
- MIT license: [LICENSE.txt](LICENSE.txt)
- Third-party software notices: [NOTICE](NOTICE)

## Privacy

MtReNIRS does not implement project-specific telemetry or transmit research
inputs and outputs to its maintainers. The optional Streamlit interface may
send framework-level usage statistics under Streamlit's default configuration.
See [PRIVACY.md](PRIVACY.md) for scope, local and remote deployment behavior,
and opt-out instructions.
