# MtReNIRS: Montage-aligned Transferable Representations for fNIRS

**Last Updated:** 31-AUG-2026

# Model summary

| **Developer** | **Microsoft Corporation** |
| --- | --- |
| **Description** | MtReNIRS is a self-supervised foundation model for
 functional near-infrared spectroscopy (fNIRS). It learns transferable
 neurovascular representations from heterogeneous public fNIRS datasets using
 masked-reconstruction pretraining. This release provides the pretrained
 inference encoder, model weights, documentation, and the MtSi1005
 montage-alignment toolkit; it does not distribute training datasets or
 participant-level recordings. |
| **Model architecture** | Six-layer Transformer masked autoencoder with signal and spatial/chromophore embeddings and temporal rotary position encoding. |
| **Parameters** | 1–500M active parameters; the released inference encoder
 contains 1,226,368 parameters. |
| **Inputs** | Preprocessed continuous oxygenated hemoglobin (HbO)
 and/or deoxygenated hemoglobin (HbR) fNIRS measurements sampled at 10 Hz,
 with channel coordinates mapped to the Standard 10-05 space. The primary training configuration used at most 256 measurement channels per crop. The public inference API accepts variable channel counts; inputs above 256 channels are outside the training and evaluation envelope. The input is expected to comprose of 10-second
 windows. |
| **Outputs** | Continuous latent embeddings for downstream research
 tasks such as linear probing, transfer learning, few-shot calibration, and
 user-defined classification. |
| **Context length** | For the primary configuration, up to 1,536 measurement
 tokens per 60-second crop during pretraining; public inference uses 10-second
 windows. |
| **Training Dates** | July-August 2026 |
| **Release date**<br>**Release date in the EU (if different)** | Expected September 2026; expected EU release date: same as
 general release unless otherwise specified.*[[BB1]](#_msocom_1)* |
| **License** | Open source under the MIT License. |
| **Model dependencies:** | N/A. MtReNIRS was trained from random initialization and
 is not a modification or fine-tune of a previously released foundation or
 frontier model. |
| **List and link to any additional related
 assets** | MtSi1005 montage-alignment toolkit, model inference code,
 pretrained encoder weights, evaluation documentation, and reproducibility
 guidance are released in the same GitHub repository under the MIT License.
 Training data, raw recordings, training code, and participant-level datasets
 are not included. |
| **Acceptable use policy** | N/A. Use is governed by the repository license and the
 limitations and responsible-use guidance in this model card. |

# 1. Model overview

MtReNIRS is a first-party self-supervised fNIRS
 representation model trained from random initialization using a visible-token
 masked autoencoder objective. The released encoder combines a linear 10-second
 signal projection with Fourier features of Standard 10-05 coordinates and
 chromophore identity, followed by six Transformer blocks with temporal rotary
 position encoding and per-head query/key normalization. Pretraining used 12
 heterogeneous public fNIRS datasets and dataset-balanced random sampling; the released
 artifact contains the inference encoder only, with the decoder, optimizer, and
 scheduler removed.

MtReNIRS differs from fixed-layout or single-dataset
 approaches by providing a shared spatial vocabulary across heterogeneous
 montages through MtSi1005 and by explicitly representing both channel location
 and chromophore identity: oxygenated hemoglobin (HbO) and deoxygenated
 hemoglobin (HbR).

## 1.1 Key model capabilities

- Produces reusable latent representations from preprocessed oxygenated
 hemoglobin (HbO) and/or deoxygenated hemoglobin (HbR) fNIRS recordings.

- Supports heterogeneous channel layouts mapped into a shared
 Standard 10-05 coordinate system.

- Enables frozen-feature linear probing, few-shot calibration, and
 transfer learning.

- Supports low-channel and low-label research evaluations.

- Provides embeddings rather than clinical conclusions or
 autonomous decisions.

# 2. Usage

## 2.1 Primary use cases

MtReNIRS is intended for research in fNIRS representation
 learning, brain-computer interfaces, computational neuroscience, transfer
 learning, and downstream classification benchmarking. Researchers may use the
 pretrained encoder as a frozen feature extractor, attach a user-defined
 prediction head, or adapt the model in controlled experimental settings.

The model is most appropriate for non-clinical analysis of
 preprocessed fNIRS signals with known acquisition metadata and channel
 geometry. Users remain responsible for validating signal preprocessing, montage
 mapping, labels, evaluation design, and the suitability of any downstream
 application.

## 2.2 Out-of-scope use cases

MtReNIRS is not designed or validated for medical diagnosis,
 patient monitoring, treatment selection, emergency response, biometric
 identification, employment or insurance decisions, assessment of cognitive
 ability or personality, or other high-impact decisions about individuals. It
 must not be used to infer health status, mental state, intent, or protected
 characteristics without an independently validated purpose, appropriate
 consent, and applicable legal and ethical review.

The model is also out of scope for direct use on raw
 intensity data without an appropriate optical preprocessing pipeline, for
 montages lacking trustworthy channel geometry, or for devices, populations,
 cortical regions, and tasks that have not been independently validated.

## 2.3 Distribution channels

Some of MtReNIRS’s
 distribution channels include:

- Public access through an open-source GitHub repository

MtReNIRS
 is distributed as a private release candidate and, after approval, as an
 open-source GitHub repository containing model weights, inference code,
 MtSi1005, and documentation. No hosted API or Microsoft-operated inference
 endpoint is provided.

## 2.4 Technical requirements and integration guidance

MtReNIRS requires Python and a compatible PyTorch runtime.
 The released checkpoint is compatible with the supported PyTorch runtime declared by the package; users should select a PyTorch build appropriate for their hardware. CPU inference is possible for small workloads, while a CUDA-capable
 GPU is recommended for larger studies.

Inputs must be preprocessed HbO/HbR signals at 10 Hz with
 channel geometry mapped to the Standard 10-05 coordinate system. The public package provides optional generic preprocessing for supported raw fNIRS inputs, including optical-density conversion, TDDR repair, modified Beer–Lambert conversion, filtering, resampling, and run normalization. It does not reproduce the complete dataset-specific training pipeline, including all scalp-coupling screening, channel-quality handling, and montage-specific exceptions. Users must validate preprocessing for their own data. MtReNIRS should be integrated only into
 research applications with human review and independent downstream validation,
 not autonomous or consequential decision systems.

## 2.5 Supported languages

N/A. MtReNIRS operates on numerical fNIRS time-series data
 and has no natural-language interface.

## 2.6 Responsible AI considerations

Developers should treat MtReNIRS outputs as research
 representations rather than measurements of health, cognition, or intent.
 Downstream systems require independent validation for the target device,
 population, montage, task, and preprocessing pipeline; meaningful consent,
 data-governance controls, and domain-expert review should be used when human
 data are involved. High-impact or clinical use is unsupported. Users should
 document model version, preprocessing, montage mapping, evaluation design, and
 failure handling, and should monitor for dataset shift and subgroup performance
 where appropriate.

## 2.7 Known limitations

- Performance is dataset-dependent and varies across tasks,
 populations, devices, montages, preprocessing pipelines, and adaptation
 strategies.

- Cross-subject transfer was competitive but did not show uniform
 superiority over Scratch-FT or Stat-SVM; the leading adaptation method differed
 by dataset.

- The strongest evidence is within-subject few-shot calibration,
 but Shader 2021 improvements over Scratch-FT did not reach significance under
 the prespecified unadjusted tests.

- The encoder itself does not perform raw-light conversion, motion repair, physiological artifact rejection, signal-quality assessment, or clinical interpretation. The optional package preprocessing helpers provide a generic subset of these preprocessing operations but not the complete training pipeline.

- Training data were heterogeneous public research datasets and may
 not represent all demographic groups, clinical conditions, devices, or
 real-world environments.

- Reported p-values are unadjusted for multiple comparisons;
 small-subject benchmarks have limited inferential resolution.

# 3. Quality and performance evaluation

**Evidence provenance.** Detailed experimental protocols, ablation studies, subject-level statistical analyses, and comparisons with prior work are reported in the accompanying journal manuscript. The repository provides the released model, inference implementation, and supporting documentation; it does not include all training data or experimental result artifacts. The manuscript citation will be added when publicly available.

MtReNIRS was evaluated on three held-out public fNIRS
 datasets that were excluded from pretraining: Gunther 2014 (eight subjects;
 mental arithmetic versus rest), Shader 2021 (eight subjects; audio, control,
 and video), and SujinBak 2019 (30 subjects; foot, left-hand, and right-hand
 motor tasks). The primary metric was subject-level macro-F1.

In cross-subject leave-one-subject-out evaluation, the
 highest MtReNIRS point estimates were 0.873±0.078 on Gunther using a frozen
 linear probe, 0.473±0.056 on Shader using full fine-tuning, and 0.703±0.186 on
 SujinBak using LoRA. The leading transfer strategy therefore depended on the
 dataset. Frozen linear probing significantly exceeded CNN and LSTM on Gunther
 and SujinBak and exceeded LSTM on Shader under prespecified unadjusted paired
 tests, but was not uniformly superior to MLP probing, pretrained full
 fine-tuning, Scratch-FT, or Stat-SVM.

The strongest evidence for transferable representations came
 from within-subject few-shot calibration. Across two, four, six, and eight
 labeled examples per class, the pretrained frozen linear probe had the highest
 point estimate among tested methods on all three datasets. Improvements over
 Scratch-FT were statistically significant at every shot on Gunther and
 SujinBak; Shader showed the same direction without reaching significance. A
 complementary SujinBak channel-reduction study also found the pretrained model
 retained higher median macro-F1 than tested baselines as channels were reduced,
 but this single-dataset analysis does not establish general robustness across
 cortical regions or tasks.

## 3.1 Benchmark results: cross-subject transfer

| **Mean ± population standard deviation of subject-level macro-F1
 in leave-one-subject-out evaluation.** |  |  |  |
| --- | --- | --- | --- |
| **Method** | **Gunther 2014 (n=8)** | **Shader 2021 (n=8)** | **SujinBak 2019 (n=30)** |
| **MtReNIRS
 LP** | **0.873 ±
 0.078** | 0.445 ± 0.094 | 0.679 ± 0.177 |
| **MtReNIRS MLP probe** | 0.865 ± 0.077 | 0.447 ± 0.103 | 0.672 ± 0.186 |
| **MtReNIRS
 LoRA** | 0.862 ± 0.052 | 0.437 ± 0.116 | **0.703 ±
 0.186** |
| **MtReNIRS Full-FT** | 0.838 ± 0.078 | **0.473 ± 0.056** | 0.702 ± 0.191 |
| **Scratch-FT** | 0.823 ± 0.077 | 0.402 ± 0.062 | 0.667 ± 0.162 |
| **Stat-SVM** | 0.838 ± 0.074 | 0.372 ± 0.079 | 0.663 ± 0.182 |
| **CNN** | 0.744 ± 0.096 | 0.344 ± 0.064 | 0.566 ± 0.159 |
| **LSTM** | 0.643 ± 0.143 | 0.355 ± 0.081 | 0.598 ± 0.161 |

Boldface marks the highest point estimate within each
 dataset and does not imply statistical significance. The leading adaptation
 method differs across datasets.

## 3.2 Benchmark results: within-subject few-shot calibration

| **Pretrained frozen linear-probe macro-F1, averaged over 50
 sampling seeds within each subject and then summarized across subjects.** |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Dataset** | **2-shot** | **4-shot** | **6-shot** | **8-shot** |
| **Gunther
 2014** | 0.778 ± 0.116 | 0.844 ± 0.095 | 0.875 ± 0.080 | 0.896 ± 0.070 |
| **Shader 2021** | 0.391 ± 0.070 | 0.430 ± 0.097 | 0.442 ± 0.119 | 0.431 ± 0.121 |
| **SujinBak
 2019** | 0.404 ± 0.081 | 0.462 ± 0.119 | 0.500 ± 0.142 | 0.530 ± 0.151 |

Across all three datasets and all tested shot counts, the
 pretrained linear probe had the highest point estimate among the evaluated
 methods. A complementary SujinBak channel-reduction study evaluated the pretrained model and tested baselines across progressively reduced bilateral channel subsets. The result should be interpreted as a single-dataset robustness analysis..

## 3.3 Contextual comparison with prior work

| **Gunther 2014 mental-arithmetic-versus-rest few-shot accuracy
 reported by this work and prior studies.** |  |  |  |  |
| --- | --- | --- | --- | --- |
| **Pretrained model** | **2-shot** | **4-shot** | **6-shot** | **8-shot** |
| **MtReNIRS
 (10-s linear probe)** | **0.78 ±
 0.11** | **0.85 ±
 0.09** | **0.88 ±
 0.08** | **0.90 ±
 0.07** |
| **MAE** | 0.67 ± 0.26 | 0.67 ± 0.12 | 0.75 ± 0.10 | 0.75 ± 0.10 |
| **SSCL** | 0.70 ± 0.12 | 0.73 ± 0.13 | 0.69 ± 0.22 | 0.69 ± 0.22 |
| **SleepFM** | 0.75 ± 0.06 | 0.62 ± 0.09 | 0.60 ± 0.07 | 0.58 ± 0.13 |
| **EFRM** | 0.67 ± 0.26 | 0.67 ± 0.12 | 0.75 ± 0.10 | 0.75 ± 0.10 |

MtReNIRS has the highest reported point estimate at every
 matched 2–8-shot setting. This comparison is contextual rather than a
 controlled head-to-head evaluation: MtReNIRS uses a 10-second configuration,
 EFRM uses a 2-second patch, and preprocessing, sampling, evaluation, and
 uncertainty definitions may differ across studies. The prior-work values are
 reproduced from the comparison reported by Jung et al.

**Fully supervised context.** fNIRS-T reported accuracies
 of 0.54 ± 0.15, 0.58 ± 0.13, 0.69 ± 0.10, and 0.88 ± 0.09 at 6, 12, 18, and 24
 shots, respectively. Because the shot counts do not match the 2–8-shot
 pretrained-model block, these values should not be treated as a direct ranking.

## 3.4 Benchmarking methodology

Each held-out benchmark represents a distinct transfer
 setting:

- **Gunther 2014 mental arithmetic.** A binary classification
 benchmark distinguishing mental-arithmetic trials from rest in a prefrontal
 fNIRS dataset. Evaluation includes eight subjects, 29 runs, 102 common HbO/HbR
 measurement channels, and 348 exact 10-second task epochs.

- **Shader 2021 audiovisual stimulation.** A three-way
 classification benchmark distinguishing audio, control, and video conditions.
 Evaluation includes eight subjects, one run per subject, 62 common measurement
 channels, and 368 exact 10-second task epochs.

- **SujinBak 2019 motor tapping.** A three-way classification
 benchmark distinguishing foot, left-hand, and right-hand motor conditions over
 bilateral sensorimotor cortex. Evaluation includes 30 subjects, 40 common
 measurement channels, and 2,250 exact 10-second task epochs.

These datasets were held out from pretraining. Together,
 they test transfer across cognitive, audiovisual, and motor paradigms, across
 binary and three-class settings, and across small- and moderate-sized subject
 cohorts.

Cross-subject evaluation used leave-one-subject-out folds.
 Neural methods used only non-test subjects, with a subject-level inner
 validation split for epoch selection; Stat-SVM used all non-test subjects.
 Within-subject few-shot evaluation sampled exactly two, four, six, or eight
 labeled trials per class from each subject, repeated across 50 sampling seeds,
 and evaluated on the remaining trials from that subject. Seed-level metrics
 were averaged within each subject before calculating the across-subject mean
 and population standard deviation.

Paired effects were computed at the subject level.
 Confidence intervals used 20,000 subject-bootstrap replicates with seed 1337,
 and two-sided Wilcoxon signed-rank tests were unadjusted for multiple
 comparisons. Comparisons with prior published methods are contextual rather
 than controlled because preprocessing, sampling, patch duration, and
 uncertainty definitions may differ.

## 3.5 Safety evaluation and red-teaming

MtReNIRS is a non-generative research model and does not
 produce natural-language, image, audio, or video content. Conventional
 generative-model red-teaming categories such as jailbreaks, harmful content
 generation, and prompt injection are not applicable.

Pre-release risk review focused on misuse boundaries, data
 exclusion, reproducibility, signal and montage validity, model-file provenance,
 and failure modes relevant to research use. The release excludes
 participant-level recordings and training datasets. The repository includes a CodeQL scanning workflow. Additional repository-level dependency and secret-scanning controls should be confirmed through the hosting configuration. Users must independently assess
 risks introduced by downstream heads, deployment contexts, and any
 interpretation of model outputs about individuals.

# 4. Data overview

## 4.1 Training, testing, and validation datasets

MtReNIRS was pretrained on 12 publicly available fNIRS
 research datasets spanning cognitive, motor, auditory, emotion, attention,
 anesthesia, neurofeedback, and driving paradigms. The actual primary training profile contained 266 subject-dataset entries and 1,387 runs; two catalogued Chen 2023 profiles were not included in the final profile because they failed during processing. The reproducibility pipeline specifies 12 datasets and expects 1,387 run-level files. Internal aggregate preprocessing records report 203,420 non-overlapping 10-second windows, approximately 565.1 hours.
 
Training data consisted of fNIRS time-series measurements
 rather than text, images, audio, or video. The primary profile contained
 203,420 non-overlapping 10-second windows sampled at 10 Hz. Inputs included oxygenated
 hemoglobin (HbO) and deoxygenated hemoglobin (HbR) concentration signals, with
 HbO-only support for one dataset.

**Sources.** All training datasets were publicly
 available research datasets obtained from external providers. No proprietary
 Microsoft dataset, private third-party dataset, synthetic AI-generated data, or
 participant-level recording collected by this release was used. No training
 dataset is redistributed with the model.

**Selection rationale.** Datasets were selected to
 broaden coverage across devices, montages, cortical regions, tasks, signal
 representations, and study populations, supporting research on transferable
 fNIRS representations. Public availability also supports scientific reproducibility,
 subject to the original providers' terms and licenses.

**Preprocessing.** Raw continuous-wave amplitude data
 were converted to optical density, screened using scalp-coupling information,
 repaired using TDDR, converted to hemoglobin concentration using the modified
 Beer-Lambert law, filtered, resampled to 10 Hz, normalized by run, and
 windowed. Inputs already in optical-density or concentration form followed
 modality-appropriate abbreviated paths. Dataset-specific geometry and
 quality-control exceptions were documented and audited.

**Ongoing collection.** No. The released model is a
 fixed, separately versioned checkpoint and is not continuously updated with
 newly collected data.

**Personal data statement.** Microsoft follows applicable
 laws and best practices pertaining to personal data. The open-source release
 contains no raw fNIRS recordings, participant-level datasets, or covered data.

**Known data limitations.** Dataset metadata, acquisition
 quality, demographics, coordinate fidelity, and task coverage vary across
 sources. Exact source URL or version metadata for Chen 2023, Shin 2017, and
 Spape 2023 should be verified against the source publications before any independent
 raw-data reconstruction.[[BB2]](#_msocom_2)

**4.1.1 Size of dataset and characteristics**

**4.1.2.A Text training data size:** Not applicable. Text data is not part of the training data.

**4.1.2.B Text training data content:** Not applicable. Text data is not part of the training data.

*Instructions: Answer with a brief, professional
 free text response. Briefly list the main categories of text data present in
 the training data (e.g. fiction and nonfiction text, scientific text, press
 publications, legal and official documents, social media comments, source
 code). If this type of data is not part of the training data, answer, “Not
 applicable.”*

**4.1.2.C Image training data size:** Not applicable. Images are not part of the training data.

**4.1.2.D Image training data content:** Not applicable. Images are not part of the training data.

*[Instructions: Answer with a brief, professional
 free text response. Briefly list the main categories of image data present in
 the training data (e.g. photography, visual art works, infographics, social
 media images, logos, brands). If this type of data is not part of the training
 data, answer, “Not applicable.”]*

**4.1.3.A Audio training data size:** Not applicable. Audio data is not part of the training data.

**4.1.3.B Audio training data content:** Not applicable. Audio data is not part of the training data.

*[Instructions: Answer with a brief, professional
 free text response. Briefly list the main categories of audio data present in
 the training data (e.g., musical compositions and recordings, audiobooks, radio
 shows and podcasts, private audio communication). If this type of data is not
 part of the training data, answer, “Not applicable.”]*

**4.1.4.A Video training data size:** Not applicable. Video data is not part of the training data.

**4.1.4.B Video training data content:** Not applicable. Video data is not part of the training data.

*[Instructions: Answer with a brief, professional
 free text response. Briefly list the main categories of video data present in
 the training data (e.g., music videos, films, TV programs, performances, video
 games, video clips, journalistic videos, social media videos). If this type of
 data is not part of the training data, answer, “Not applicable.”]*

**4.1.5.A Other training data size:** The primary fNIRS training profile contained 203,420 non-overlapping 10-second windows sampled at 10 Hz, equivalent to approximately 565.1 hours of signal data across 266 subject-dataset entries and 1,387 runs.

*[Instructions: If other modalities are present in
 the training data, answer this question with a brief, professional free text
 response. If not, answer, “Not applicable.” Briefly describe each additional
 modality present in the training data. For each modality, describe the
 approximate size.]*

**4.1.5.B Other training data content:** Numerical fNIRS time-series measurements of oxygenated hemoglobin (HbO) and deoxygenated hemoglobin (HbR) concentration signals, spanning cognitive, motor, auditory, emotion, attention, anesthesia, neurofeedback, and driving research paradigms.

*[Instructions: If other modalities are present in
 the training data, answer this question with a brief, professional free text
 response. If not, answer, “Not applicable.” Briefly describe the main
 categories of content present in the training data for this modality. See the
 rows above for examples of the appropriate level of granularity.]*

**4.1.6 Latest date of data (acquisition/collection for model training):** August 2026

*[Instructions: Answer in DD-MMM-YYYY format, where
 month is the abbreviated month. Example date: 01-Jan-2025. Indicate the latest
 date when Microsoft collected/obtained the data for the model training. If the
 model is continuously trained on new or dynamic data after this date, say that
 it collection is ongoing instead of listing a specific date.]*

**4.1.7 Is data collection ongoing to update the model with new data collection after deployment?** No.

**4.1.8 Date the training dataset was first used to train the model:** Jul 2026

*[Instructions: Use MMM YYYY format, where month is
 the abbreviated month. Example: Jan 2025. Enter the first month and year that
 the training dataset was used to train the model. If the training dataset
 changed over time, enter the earliest month and year that model training began
 using any part of the dataset.]*

**4.1.9 Rationale or purpose of data selection:** The datasets were selected to broaden coverage across devices, montages, cortical regions, tasks, signal representations, and study populations, supporting research on transferable fNIRS representations. Their public availability also supports scientific reproducibility, subject to the original providers’ terms and licenses.

*[Instructions: Answer with a brief, professional
 free text response. In 1–3 sentences, explain why these datasets were chosen
 and how they support the model’s intended use. For example: “Datasets were
 selected to maximize coverage of conversational English and technical domains,
 supporting the model’s use in enterprise productivity scenarios.” You may find
 it helpful to describe any specific characteristics of the dataset that were
 unique (if that was the case) and made it valuable. For example, the dataset
 represented a group/topic/concept not well represented in other
 datasets. Use a professional, objective tone. Common reasons you
 can use include but are not limited to:*

- *This
 dataset was identified as the most appropriate publicly available source
 for [specific purpose]. While the dataset has limitations, at the time of
 use, it represented the best option among existing alternatives.*

- *Publicly
 available dataset was selected as it allows for replication of
 scientific findings*

- *Dataset
 reflects new datatypes that accelerate innovation in model design or
 capabilities]*

4.2 List of data sources

*[Instructions: In this section “dataset” should be
 understood as a single, pre-packaged collection of data. The filtering and pre-
 processing of data collected from the same pre-packaged collection should not
 be considered a new dataset to be disclosed separately in the sections below.
 If a particular dataset can be assigned to more than one of the categories
 below, providers should select the most relevant category and only report the
 dataset in that category, except in the case of synthetic data.]*

4.2.1 Publicly available datasets

**4.2.2 Have you used publicly available datasets to train the model?** Yes.

4.2.2 Private non-publicly available datasets
 obtained from third parties

*4.2.2.1 Datasets commercially licensed by
 rightsholders or their representatives*

**4.2.2.1A Have you concluded transactional commercial licensing agreement(s) with rightsholder(s) or with their representatives?** Not applicable.

*4.2.2.2 Private datasets obtained from other third
 parties*

**4.2.2.2.A Have you obtained private datasets from third parties that are not licensed as described in Section 2.2.1, such as data obtained from providers of private databases, or data intermediaries?** No.

4.3 Personal Data

**4.3.1 Was personal data used to train the model?** Microsoft
 follows applicable laws and best practices pertaining to personal data.

*[Instructions: Do not modify this text]*

4.4 Synthetic data

**4.4.1 Was any synthetic AI-generated data used to train the model?** No.

4.4 Data processing aspects

4.4.1 Respect of reservation of rights from text
 and data mining exception or limitation

**4.4.1A Does this dataset include any data protected
 by copyright, trademark, or patent?** Microsoft
 follows applicable laws and best practices for processing data protected by
 copyright, trademark, or patent.

*[Instructions: Do not modify this text]*

4.4.2 Other information

**4.4.2A Does the dataset include information about
 consumer groups without revealing individual consumer identities?** Microsoft
 follows applicable laws and best practices for protecting consumer
 identities.

*[Instructions: Do not modify this text]*

**4.4.2B Was the dataset cleaned or modified before model training?** Yes.

# 5. Contact

Requests for additional information can be directed to [MSFTAIActRequest@microsoft.com](mailto:MSFTAIActRequest@microsoft.com).

**Authorized representative:** Microsoft Ireland Operations Limited 70 Sir John
 Rogerson’s Quay, Dublin 2, D02 R296, Ireland

# 6. Appendix

Detailed architecture, preprocessing, benchmark,
 statistical, and reproducibility specifications accompany the repository. Key
 reproducibility facts include a six-layer, 128-dimensional Transformer encoder;
 25% random masking; a 10-second primary token configuration; dataset-balanced
 pretraining; subject-level evaluation. Within the selected training run, the released epoch checkpoint was selected using self-supervised validation reconstruction loss. Selection among candidate preprocessing and model configurations also considered downstream transfer evaluation.

[[BB1]](#_msoanchor_1)Reminder
 to update when you go to publish. Just one date is sufficient here if it’s the
 same for EU and elsewhere (common).

[[BB2]](#_msoanchor_2)The
 template calls for just a high-level paragraph here, and then the other details
 Tony provided should be distributed across section 4, where requested by the
 template. Sorry for the nuisance and thank you!
