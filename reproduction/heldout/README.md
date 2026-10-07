# Held-out test analysis

This directory contains the aggregate analysis of the fixed seven-video-group test split, separate from validation and the external evaluation/selection sets. The primary comparison uses the three epoch-99 checkpoints. RGB epoch 73, grayscale epoch 58 and signed-residual epoch 53 are secondary checkpoints selected using the external sets. This analysis never selects a checkpoint or fits an operating threshold to the test scores.

The frozen test cohort contains **6,592 video frames** (3,296 per class) from **seven video groups**, plus **4,000 synthetic photo recipes** (2,000 originals, each with a sharp and a blurred recipe). Video groups were split before frame selection. The labels measure agreement with a filtered frozen teacher; photo sharp labels are construction labels. Neither is an independent human reference for objective blur accuracy.

Read [RESULTS.md](RESULTS.md) for the tables and [heldout_results.json](heldout_results.json) for full aggregates, thresholds and input hashes. The [execution record](EXECUTION.md) and [sanitised protocol](execution_protocol.json) identify the completed
run, including the byte-exact [executed v1 source](executed_runner_v1.py). The original private score files and source
images are not distributed.

## Fixed protocol

Each checkpoint uses its own r90, r95 and r98 thresholds derived from its saved **validation-frame scores**. `p_blurred >= threshold` is classified as blurred; a sharp sample is kept when `p_blurred < threshold`. The threshold stays unchanged for the original test cohort and every bootstrap draw. A validation recall target therefore does not promise the same realised recall on the test cohort. Both sharp retention and blurred recall are reported.

The primary AUC gives tied sharp/blurred scores half credit. All video metrics pool the frames, matching the original validation report. Because the seven groups have unequal sizes and one contributes about 64% of test frames, the report also includes:

- Within-group rates and AUC under anonymous names `video_01` through `video_07`.
- Equal-group mean sharp retention and blurred recall, giving each group the same weight.
- Seven leave-one-group-out pooled analyses, omitting each complete group in turn while preserving the threshold.
- Paired differences between the three primary checkpoints.
- The measurement-agreement subset, when its frozen membership file is supplied. Subset labels must match the index labels; agreement with the classical measurement does not establish independent human ground truth.

For each of **2,000 bootstrap draws**, seven groups are sampled uniformly with replacement and their frames are repeated together. The same draw is used for all six checkpoints and all video metrics, allowing paired comparisons. Percentile intervals use the 2.5th and 97.5th percentiles; the random seed is **20261007**. The bootstrap draws do not recalibrate thresholds. If a measurement-agreement subset draw has only one class, its AUC is undefined and excluded from the interval; the JSON records the number of valid draws for every metric.

Photo uncertainty resamples 2,000 original-photo pairs with replacement, retaining both recipes from each drawn original. The same original-pair draws are used for every checkpoint. This preserves sharp/blurred pairing but does not capture dependence between different originals in the same source album. Photo results are secondary synthetic-recipe measurements.

These intervals condition on fixed weights, one training run per representation, the chosen split and the available labels. They do not estimate variation across training seeds, teacher error, unrecognised duplicate sources or deployment domains. Seven clusters provide limited information about population uncertainty; percentile intervals may have poor coverage. Pairwise intervals are descriptive and do not constitute equivalence tests or corrected multiple comparisons.

The term *held-out* describes separation from training and validation. A repeat evaluation, or results inspected before a protocol was fixed, must not be called an untouched prospective test. Execution and any earlier use should be disclosed alongside the results.

## Reproduce the aggregate analysis

The separate [run_test.py](run_test.py) prepares a private cache from the frozen index and deterministic evaluation
recipes, then scores all six public weight files through a pinned copy of the trainer's historical GPU-autocast
inference path. It hashes the current originals, mask inventory and prepared arrays, guards resumes with a manifest,
retains incomplete shape-bucket tails, and writes probabilities with 17 significant digits. Input loading uses threads
with a main guard to avoid the earlier Windows process-spawn stall. Preparation is CPU-only; scoring needs an available
GPU and the original trainer's dependencies. All cache and per-image outputs belong in a private working directory.

```text
python reproduction/heldout/run_test.py prepare --trainer-root private/trainer --weights weights --out work/heldout-run
python reproduction/heldout/run_test.py run --out work/heldout-run --batch 8 --memory-fraction 0.70
```

Before any test prediction, the scorer checks the public and trainer implementations on existing openly licensed
demonstrations and preflights memory with synthetic zero arrays at the largest prepared geometry. It then locks the
device, precision, batch size, comparator and memory limit. A changed source, cache, weight or locked parameter is
rejected. The earlier abandoned script had already read test metadata; these records cannot establish historical
non-use or retroactive preregistration.

Requires Python 3.10 or newer and NumPy. No PyTorch, GPU, model weights or image reads are needed for this analysis step. Supply the private frozen index, six full-precision score CSVs and the inference manifest:

```text
python reproduction/heldout/analyse_test.py --index private/index.csv --scores private/scores --manifest private/run_manifest.json --confidence private/confident_frames.csv --out reproduction/heldout
```

Omit `--confidence` when that frozen subset file is unavailable. The script refuses a cohort with different counts, groups, class labels or sample IDs. It verifies the private index SHA-256 against the inference manifest. It requires locked prediction parameters and a complete `.json` sidecar for every CSV, checking the actual
manifest, score-file and checkpoint hashes and the exact sample count. Public output contains hashes and anonymous group aggregates; it contains no sample identifiers, video names, source paths or raw probabilities.

The index must have `sample_id`, `source`, `split`, `target`, `group` and `png`. Sources must be `frames` or `photo`; each original photo path must identify exactly one sharp and one blurred recipe. A score file is named after its public checkpoint ID and must contain `sample_id`, `source`, `target` and `p_blurred`. Probability values are read as Python double precision; preserving 17 significant digits avoids intentional six-decimal rounding in the earlier score exports.

The manifest has `prediction_parameters_locked: true`, `index_sha256` and a `checkpoints` array with six entries, each containing `id`, `input`, `epoch`, `weights_sha256`, `thresholds` (r90/r95/r98) and `selection_role`. The accepted IDs are `rgb_ep099`, `gray_ep099`, `p99_ep099`, `rgb_ep073`, `gray_ep058` and `p99_ep053`. The three epoch-99 checkpoints are `primary`; the other three are `secondary`. Threshold source and inference-cache provenance must be established before running this analysis; matching sample IDs alone cannot establish which weights generated a file.

An independent mathematical self-check can run without any private input:

```text
python reproduction/heldout/analyse_test.py --self-test
```

It compares the grouped AUC, the weighted tied-score AUC and fixed-threshold rates with direct expanded-array calculations, covering tied scores, omitted groups and repeated groups.
