# Validation sampling uncertainty

These intervals were computed from the three saved **epoch-99 validation prediction files**, without running inference,
reading images or evaluating the held-out test split. The cohort contains 6,602 frames: 3,301 sharp and 3,301 blurred
frames from 11 video groups. Every group contributes equally many sharp and blurred frames, although group sizes differ.

The reference labels are the filtered frozen teacher's labels. All metrics therefore measure teacher agreement,
not objective blur accuracy. The point estimates exactly reproduce the published epoch-99 validation AUC and
sharp-kept values and the exported weight metadata.

## Method

For each of 5,000 bootstrap draws, 11 video groups are sampled uniformly with replacement. All frames of each drawn
group enter together; a repeated group contributes its frames repeatedly. The same draw is used for all three
models, enabling paired model comparisons. Metrics pool the frames, as in the original report; this is not a mean
of per-video metrics. Intervals are the 2.5th and 97.5th percentiles. The random seed is 20261007.

Two operating-point analyses are reported:

- **Fixed threshold:** keep the original model-specific validation threshold in every draw. Both the share of sharp
  frames kept and the realised blurred recall vary. These intervals condition on the original threshold and do not
  include uncertainty from estimating a new threshold.
- **Recalibrated threshold:** within each draw, recompute the blurred-score quantile for the chosen recall target,
  then measure sharp frames kept in that same draw. This resamples the calibration step jointly with the frames.
  The nominal recall is imposed within the bootstrap draw; it is not a recall guarantee on new videos. With finite
  samples, interpolation and tied scores, realised recall need not equal the nominal target exactly.

The AUC calculation gives tied positive/negative scores half credit. The script checks its grouped pair-count
calculation and its weighted linear quantile against direct expanded-array examples with ties, omitted groups and
repeated groups before analysing the saved data. Private identifiers are used only in memory; outputs contain
aggregate metrics and input hashes, without sample IDs, video names or local source paths.

## Results

The table shows point estimates and 95% intervals. The sharp-kept columns use **recalibrated thresholds**.

| Input | Frame AUC | Sharp kept at 95% recall target | Sharp kept at 98% recall target |
|---|---|---|---|
| RGB | 0.952 [0.924, 0.974] | 74.9% [61.8, 86.5] | 61.7% [49.7, 74.1] |
| Grayscale | 0.942 [0.904, 0.973] | 70.0% [61.8, 87.5] | 52.1% [44.9, 75.6] |
| Signed residual | 0.949 [0.927, 0.973] | 73.9% [69.7, 85.2] | 58.2% [50.6, 73.2] |

At the fixed original 95% threshold:

| Input | Sharp kept, 95% interval | Realised blurred recall, 95% interval |
|---|---|---|
| RGB | 65.3–81.2% | 89.8–98.1% |
| Grayscale | 62.4–78.2% | 89.1–99.0% |
| Signed residual | 67.6–82.9% | 89.6–98.6% |

All three pairwise differences in AUC, and in sharp-kept at recalibrated 95% and 98% recall targets, have intervals
containing zero. This does not establish statistical equivalence. The complete aggregate output, including the
90% operating point, fixed-threshold intervals and paired differences, is in
[validation_epoch099_bootstrap.json](validation_epoch099_bootstrap.json).

## Limits

These are conditional intervals for **video sampling with fixed trained weights**, not uncertainty over training
seeds, model initialisation, teacher mistakes, source-selection bias or the choice of split. Only one run per input
was trained, and only 11 validation groups are available; percentile intervals may have limited coverage. Validation
was repeatedly used during development, so these are not untouched test results. Generalisation to other domains
and equivalence of the three input representations remain unproved.

The saved sigmoid scores have six decimal places. Quantisation can create ties and alter metrics relative to a future
calculation from full-precision predictions. The recorded input hashes fix exactly which saved scores were used here.

## Reproduce

Requires Python and NumPy; no PyTorch, model weights, GPU or source images are required. Supply the private frozen
training index and the three saved validation CSVs. The index needs `sample_id`, `source`, `split`, `target`, `group`;
the prediction files need `sample_id`, `source`, `target`, `p_blurred`. Photo rows are ignored. Frame IDs must match
the validation cohort exactly, preventing accidental substitution of training or held-out scores.

```text
python reproduction/uncertainty/bootstrap_validation.py --index private/index.csv --predictions rgb=private/rgb_val_epoch_099.csv --predictions gray=private/gray_val_epoch_099.csv --predictions p99=private/p99_val_epoch_099.csv --out reproduction/uncertainty
```

The private score files are not distributed. Anyone with those files can reproduce the aggregates and verify their
hashes; the public script alone cannot reconstruct the predictions. Do not publish the private inputs with this report.
