# Aggregate evidence and reproducibility

This package exposes the arithmetic behind the exploratory codec choice and the fixed study's sample counts. It contains
no source images, private paths, private photo/video names, private frame identifiers or per-image predictions. The three
video labels below are anonymous group aliases. The codec/split/provenance checks use saved local records. The
separate [held-out package](heldout/README.md) now includes one completed fixed-checkpoint evaluation and its aggregates.

Run with Python 3.10 or later; no packages, dataset, model weights or GPU are required:

```console
python reproduction/verify_evidence.py
python reproduction/verify_evidence.py --json
```

The script recalculates the codec rankings, checks the original-pool and evaluation-cap arithmetic, and checks consistency
of the released run summaries. It reads only files in this directory and writes no files. It does **not** reproduce the
private image measurements or training, and it cannot independently hash private input files.

## Codec comparison

[codec_group_medians.csv](codec_group_medians.csv) contains full-precision group medians recomputed from the original
numeric measurement records: 40 original photos, each tested as PNG, four JPEG qualities and four H.264 CRFs, versus
100 teacher-`sss` frames from each of three videos. Photos in this comparison received no injected motion blur. The codec
settings, residual and metric definitions are recorded in [codec_criterion.json](codec_criterion.json).

The comparison used the ideal, double-precision residual `d = Y − K*Y`, with replicated borders and the fixed mesh kernel.
It did not use the published detector's bfloat16-rounded fingerprint. In each image, the lowest 30% of 16-pixel cells by
mean absolute residual define the flat-region mask; the highest 10% define the strong-region mask.

| Metric | Definition |
|---|---|
| `flat_e3` | 1,000 × mean absolute residual in flat cells |
| `edge_e3` | 1,000 × mean absolute residual in strong cells |
| `flat_edge` | Per-image flat/strong ratio, then median across images |
| `zero` | Fraction of all cropped-image pixels with `abs(d) < 2e-4` |
| `block8` | Mean residual steps at eight-pixel row/column boundaries relative to steps elsewhere |
| `hcorr` | Horizontal lag-one signed-residual correlation, selecting pairs by the left pixel's flat-cell mask |

Each column is a median of per-image measurements. In particular, `flat_edge` is **not** obtained by dividing the two
displayed group medians. Incomplete 16-pixel cells at the bottom and right are discarded for measurement.

For a video's median vector `v` and a photo variant's median vector `p`, the recorded criterion is:

```text
D(v, p) = |ln(v.flat_edge / p.flat_edge)|
        + 5 |v.zero − p.zero|
        + 5 |ln(v.block8 / p.block8)|
        + 2 |v.hcorr − p.hcorr|
```

It selects H.264 CRF 23 for each recorded group:

| Video group | Nearest tested photo variant | D |
|---|---|---:|
| A | H.264 CRF 23 | 3.130084102 |
| B | H.264 CRF 23 | 1.230128519 |
| C | H.264 CRF 23 | 1.240426316 |

The full CSV precision matters: recomputing from a table rounded to four decimal places gives slightly different
distances. This is a chosen exploratory criterion, with weights 1/5/5/2, rather than a calibrated perceptual distance.
The conclusion is conditional on these sources, settings, metrics and weights; it does not establish equal noise floors,
removal of all noise, or equivalence to temporal P/B-frame coding. Frames within a video are correlated. The separate
visual observation of noise streaks after added blur is not quantified by this original-photo codec table.

## Photo originals and recipe rows

[photo_split.json](photo_split.json) resolves the difference between the material count and the training pool:

| Original pool | All originals | Train | Validation pool | Held-out test pool |
|---|---:|---:|---:|---:|
| Camera EXIF photos | 5,194 | 3,587 | 803 | 804 |
| Accepted border-cropped photos | 43,372 | 29,736 | 6,818 | 6,818 |
| Reviewed 4K frames | 231 | 161 | 35 | 35 |
| Total | 48,797 | 33,484 | 7,656 | 7,657 |

The first two rows total **48,566 camera photos**. Including the 231 reviewed 4K frames gives **48,797 anchor originals**.
Whole manifest `shoot_group` identifiers govern the first pool; whole albums govern the other two. The deterministic
group-assignment search targets approximately 70/15/15 sample shares. It assigns originals before generating sharp and
blurred recipe rows, and the index builder rejects a declared group crossing splits.

All 33,484 training originals remain in the index, each with one sharp and one blurred recipe row. Validation and test
are each capped at 2,000 originals, so each has 2,000 rows per class. The other **11,313 held-out originals** are omitted
from the capped index and are not returned to training. A local aggregate check found no original-identity or declared-group
crossing between train and validation. These controls do not prove that all visually similar material across different
groups has been discovered. Exact group assignments are private and cannot be reconstructed from these aggregates.

The frame metadata also distinguishes 37 processed files from 36 distinct video groups: the frozen selector excludes
one duplicate copy. Its summary contains 3,819 framing units and 371,743 eligible labelled frames after selection filters;
the study's 374,019 teacher-labelled frames refer to the broader count before those filters. The held-out counts in the
JSON are split metadata, not model outcomes, and this package performs no test evaluation.

## Fixed run records

[run_provenance.json](run_provenance.json) provides the shared index and selection SHA-256 digests and anonymous summaries
of the three published 100-epoch runs. The input index and frame selection still matched each run's recorded start hashes
when checked locally. Every recorded resume matched its run's starting source-code digest. RGB and residual runs share
a source-code digest; grayscale uses the later code state documented in METHOD.

The training configuration's index digest guards resumes against changes to the configuration or index, and published-run
mode rejects changed source code. Later database labelling therefore does not redefine this fixed indexed sample pool.
The public verifier checks agreement among these released records; it cannot reproduce the private-file hashing audit.
The recorded provenance also lacks a complete per-image checksum manifest proving historical pixel-file immutability.
Complete training code, configurations and data remain necessary for full training reproduction.

## Reproducible explanatory figures

The repository's English/Hungarian infographics and coding-path explanation use native SVG, generated by
[render_figures.py](render_figures.py):

```console
python reproduction/render_figures.py
```

These figures explain the method and its qualifications. The codec bars plot the recorded aggregate medians, and the
infographic's metric table reads the published epoch-99 values from README. They contain no private source-image crops
or illustrative images presented as measurements. The codec table and explicit criterion above supply the numeric data.

## Validation uncertainty and public GoPro identities

The [validation bootstrap](uncertainty/README.md) releases 5,000 paired video-group resamples of saved epoch-99 scores
from 11 validation groups, along with its [aggregate intervals](uncertainty/validation_epoch099_bootstrap.json).
It requires NumPy and private saved-score inputs to recompute; no inference or GPU is needed. The intervals condition
on one trained model per input and teacher-reference labels, and do not cover seed variation or objective accuracy.
All pairwise difference intervals for AUC and recalibrated retention at 95% and 98% recall include zero; this does not
establish equivalence.

The [GoPro provenance record](gopro/README.md) identifies the user-confirmed Kaggle mirror: 2,058 PNG images, or 1,029
blur/sharp pairs. Public mirror member names and byte checksums are supplied to identify this exact copy, without
redistributing images. These are public evaluation-file identities, separate from private training-source identifiers.
Correspondence to the official 1,111-pair test split and its blur variant is not established.

The [experiment plan and completion notes](EXPERIMENT_PLAN.md) distinguish the completed internal test from
pending human-reference, multi-seed, domain and ablation studies. [Test results](heldout/RESULTS.md) include fixed-threshold
recall, conditional group-bootstrap intervals and group sensitivity analyses. No new training was performed; the package
does not supply independently adjudicated perceptual accuracy. Validation intervals retain the scope described above.
