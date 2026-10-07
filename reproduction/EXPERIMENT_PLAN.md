# Evidence and evaluation plan

Status: proposed protocol, not executed. Creating this file does not run inference, inspect held-out predictions,
collect new human labels, or start training. It does not retroactively make the earlier exploratory work preregistered.
The current native-frame results remain agreement with retained teacher labels.

## 1. Evidence completed without a new model run

The reproduction package verifies the saved codec tables and their stated heuristic, the photo split/cap
arithmetic, and the frozen-run provenance summaries. These checks use stored numeric records and code, not source
images or held-out model results. They support the documented procedures; they do not establish perceptual accuracy,
historical pixel immutability, or a causal benefit from H.264, sign preservation, or percentile normalisation.

The native SVG summaries and reference-class captions were corrected. Counts described as
false positives or misses are relative to the stated reference labels. Teacher disagreement alone does not determine
which side is correct. No new performance numbers should be inferred from a corrected illustration.
The paired validation bootstrap was computed from existing epoch-99 scores and is released separately; its intervals
condition on fixed models and teacher-reference labels. The GoPro mirror's 2,058 PNG files were identified by byte
checksums, without image decoding. Neither check runs the held-out experiment below.

## 2. Lock a finite evaluation before accessing predictions

The proposed evaluation covers the existing seven-group internal test partition in the frozen index. The documented
frame counts are 3,296 reference-sharp and 3,296 reference-blurred rows; both native-frame classes are teacher-derived.
Verify the actual frozen frame counts and group assignments against the index before execution.
Also include the already capped synthetic-photo test partition, expected to contain 2,000 originals with one fixed
construction-sharp and one blurred recipe each. Do not add newly processed database entries or new teacher labels.

Freeze a manifest containing:

- The evaluator source hash, inference source hash, software versions, device/precision, batch size and score comparator.
- The complete six-checkpoint list, checkpoint byte hashes and unrounded validation thresholds from their metadata.
- The original index hash, selection hash, split membership, anonymised group IDs, expected counts and mask rules.
- The deterministic evaluation order and fixed photo recipes, including the synthesis-code version.
- The statistics, bootstrap seed and number of replicates specified below.
- The execution date and a signed-off access-history record stating whether test images, labels, scores or summaries
  were previously inspected or used in development. This new protocol cannot establish historical non-use by itself.

The three epoch-99 checkpoints are the primary comparisons: RGB, grayscale and p99 residual. The already published
RGB epoch 73, grayscale epoch 58 and p99 epoch 53 checkpoints are secondary comparisons selected using external
evaluation material. Evaluate and report all six. Do not choose a winner, threshold or new checkpoint from this test.

The recorded index SHA-256 is
`1d82624aad595396f1943c80cc2acf520afd655ae2cd4f168c20047d2b548810`; the frame-selection SHA-256 is
`5b2f70e6515b7b8c682d746b17e28ea9304fab20d011ae0fd63b6d394bcc7003`. A mismatch must be resolved as an input-integrity
issue before running the locked evaluation, rather than silently substituting a newly generated index.

Record current image and mask checksums privately where practical, and check exact duplicate/group overlap across
the locked partitions. These new checksums make the prospective evaluation reproducible; they do not prove that
every source pixel was unchanged throughout the historical training runs. Near-duplicate checks are a separate
integrity audit, not a reason to adjust a test selection after seeing its model scores.

## 3. Run without competing with active labelling

Keep this stage queued while the existing GPU labelling job is active. Do not stop that job, alter its configuration,
launch a second GPU workload, or silently substitute a long CPU evaluation. Use a separate writable run directory;
the source projects, database and frozen training runs remain inputs.

The local draft `tools/test_split.py` is a starting point for review, not a ready command to invoke unchanged. Its
top-level code accesses test confidence labels, creates output directories and uses CUDA and cached scores. Before
execution, prepare a standalone runner with an explicit output directory and a main guard. Check its handling of
cached predictions, thresholds and resampling without importing or evaluating the held-out data.

First verify the runner's input processing and numerical agreement on a fixed set of at most 16 existing openly
licensed demonstration images. This is an implementation check, not a performance comparison. Lock any corrections
before test inference. Then process one checkpoint at a time using the documented CUDA bfloat16 evaluation path and
fixed batch size 8. Preserve the deterministic evaluation masks from the original configuration; do not replace them
with all-valid masks or re-encode native frames. There is no test-time augmentation or per-test threshold calibration.

The planned upper bound is six checkpoints times 6,592 native frames plus 4,000 fixed photo variants: 63,552 image
scores, subject to the verified frozen counts. Save scores in chunks, preserving their full numeric precision.
A resumed job may fill missing scores for the same manifest; it must not reuse a cache with different weights,
inputs, masks, precision or evaluator code. Failed, missing or non-finite scores remain visible and block an unqualified
complete-result claim; they are not silently removed from denominators.

There is no new training or teacher labelling in this stage. The clock time is to be measured, not inferred from the
training throughput. One completed locked prediction set is enough; errors may justify an audited repair, not
checkpoint or threshold search.

## 4. Report fixed operating points and dependence-aware uncertainty

For every checkpoint, report native-frame ROC AUC and the following at its stored validation r90/r95/r98 thresholds:

- The share of reference-sharp frames retained.
- The share of reference-blurred frames detected.
- Counts and denominators, with one table for each of the seven anonymised video groups.

The 95% and 98% names identify validation targets. Test recall is measured at those locked thresholds and may differ;
do not move the thresholds until the test reaches the named recall. These are teacher-reference metrics, not
independently established human accuracy, probability calibration or universal operating guarantees.

Report pooled metrics and a clearly separate equal-group mean of the retention/recall metrics. Balancing sharp and
blurred rows within each group does not give every group equal weight in a pooled result. Keep group sizes visible.
The already specified measurement-agreement subset can be secondary, with its counts and the same caveat that
measurement agreement does not prove the label. Synthetic-photo AUC is agreement with construction labels and is
reported separately from native-frame performance.

Use 2,000 video-group bootstrap replicates with seed `20261007`: sample seven groups with replacement, keeping each
chosen group's frames together. Use the same resampled groups for all six models so their differences are paired.
Report percentile 2.5%/97.5% intervals for pooled AUC, retention, recall and the three pairwise primary-model
differences. Do not treat correlated frames as independent bootstrap units. For synthetic-photo uncertainty, resample
originals with their sharp/blurred variants together, separately from the video bootstrap.

With only seven groups these intervals are exploratory and can be unstable. They describe source-group resampling
for fixed models and fixed validation thresholds; they do not include training-seed variation, threshold-selection
uncertainty or teacher-label uncertainty. Also show all seven per-group results and a seven-case leave-one-group-out
summary as a sensitivity description. Overlapping intervals or a non-significant difference do not establish model
equivalence. No equivalence margin has been defined for this study.

Public outputs can contain anonymised group counts, checkpoint/input hashes, aggregate metrics, intervals and the
frozen protocol. Private paths, source titles, filenames, images, teacher response text and example thumbnails need
not be published. An aggregate report does not make the copyrighted training/test material publicly reproducible.

## 5. Separate follow-up studies from the existing test result

These studies remain pending; they are not started by this plan and must not reuse the seven-group test set for
tuning. Finish the fixed-model evaluation before changing the research question or training recipe.

| Question | Bounded first study | What it can support |
|---|---|---|
| Are teacher disagreements perceptually justified? | At most 220 frames from all eleven frozen validation groups: up to ten per retained teacher class per group, spread across framing units. Where fewer are available, take all eligible frames and report the shortfall. Three independent raters use fixed viewing conditions, a written usability/blur rubric and an uncertain/other-degradation option. Raters do not see model scores or teacher answers. | Human-reference agreement on this explicitly sampled material and rater variability. It does not establish universal sharpness ground truth or unbiased prevalence in all video frames. |
| Does performance survive new source domains? | A separately acquired, source-disjoint 400-image set from four predeclared domains/sources, with the same human rubric and frozen thresholds. Its inclusion rules and duplicate audit are fixed before scoring. | Evidence for those additional sources, with domain-specific results. It cannot by itself establish universality. |
| How sensitive are the three input comparisons to training randomness? | Three fresh, matched seeds for each input: nine runs using one frozen implementation, schedule, data selection and explicit interruption/sampler-state handling. | Training-seed variation under that implementation. The historical single-seed curves remain a separate result. |
| Does rounding or residual sign affect learning? | One matched three-run pilot: a newly trained rounded signed-residual baseline, an otherwise matched exact-float32 signed residual, and an otherwise matched residual-magnitude input. Keep the same seed, normalisation, masks, architecture and schedule. | Preliminary effects of those two controlled changes. A direct input swap into existing weights only measures distribution sensitivity and is not a substitute for retraining. |
| Does encoding reduce a learning shortcut? | Two matched 100-epoch pilot runs, changing only whether synthetic anchors are encoded after blur. Preselect 64 new openly licensed originals and a diagnostic grid of eight injected lengths, four directions, three prescribed added-noise levels and two coding states: 12,288 images. Use the same grid and fixed thresholds for both models; intermediate-length score curves remain descriptive. | A causal comparison under the chosen controls, with synthetic-extreme labels still defined by construction. The present exploratory codec-statistics ranking alone does not show what a trained network used. |

The multi-seed and ablation stages are substantial new compute work. The reported 13.6 minutes per epoch implies
roughly 22.7 GPU-hours for one 100-epoch run on the measured setup; it is an estimate, not a runtime guarantee.
They are therefore separate research stages, not necessary consequences of correcting the repository's wording.

The current defensible contribution remains the documented recipe and scoped empirical comparison: the published
rounded, signed mesh residual alone reaches validation discrimination close to the RGB and grayscale runs on the
examined corpus. Residual-domain learning, blur detection and VLM supervision have prior work listed in the README.
This plan adds neither a priority claim nor new evidence for statistical equivalence or universal detection.
