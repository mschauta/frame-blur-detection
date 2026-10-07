# Seven-group held-out evaluation

All checkpoints and operating thresholds were fixed before this calculation. The primary comparisons use epoch 99; earlier externally selected checkpoints are secondary. Test scores are never used to adjust thresholds or choose a checkpoint.

Frame metrics measure agreement with filtered frozen-teacher labels, not objective blur accuracy. The cohort has 6,592 frames (3,296 per class) from seven video groups and 4,000 synthetic photo recipes (2,000 paired originals). Photo sharp labels are assigned by construction and do not establish objective sharpness.

The largest video group supplies 63.8% of test frames. Pooled metrics therefore mainly describe the composition of this cohort. Equal-group operating points and seven leave-one-group-out analyses accompany the pooled estimates.

## Primary results

Values are point estimates with 95% percentile bootstrap intervals. r95 denotes a validation recall target; the realised test blurred recall is measured separately.

| Input | Frame AUC | Sharp kept at fixed r95 threshold | Blurred caught at fixed r95 threshold |
|---|---|---|---|
| rgb (epoch 99) | 0.952 [0.907, 0.961] | 68.7 [42.0, 75.2]% | 96.5 [94.3, 97.4]% |
| gray (epoch 99) | 0.953 [0.949, 0.965] | 77.3 [70.7, 82.1]% | 95.3 [94.4, 97.5]% |
| p99 (epoch 99) | 0.948 [0.919, 0.956] | 75.5 [59.7, 80.1]% | 94.3 [93.6, 96.5]% |

## Equal-group operating points

Each video group receives equal weight in this table. These are means of within-group class-specific rates, not an alternative pooled AUC.

| Input | Mean sharp kept, r95 | Mean blurred caught, r95 |
|---|---|---|
| rgb | 50.8 [26.7, 72.1]% | 94.3 [89.1, 98.2]% |
| gray | 72.3 [63.7, 80.7]% | 93.4 [88.5, 96.6]% |
| p99 | 70.3 [56.9, 81.4]% | 91.0 [80.2, 97.6]% |

## Paired primary differences

The same video-group draw is used for every checkpoint. Differences are first model minus second model; retention and recall differences are percentage points. An interval containing zero does not establish equivalence.

| Comparison | AUC difference | Sharp-kept difference, r95 | Blurred-caught difference, r95 |
|---|---|---|---|
| rgb_ep099 minus gray_ep099 | -0.002 [-0.047, 0.010] | -8.6 [-32.2, -1.0] pp | 1.2 [-2.7, 2.7] pp |
| rgb_ep099 minus p99_ep099 | 0.003 [-0.018, 0.008] | -6.8 [-20.2, 2.7] pp | 2.1 [-0.7, 3.4] pp |
| gray_ep099 minus p99_ep099 | 0.005 [-0.003, 0.034] | 1.8 [-0.3, 12.8] pp | 1.0 [-0.2, 2.1] pp |

## Secondary checkpoints and photo recipes

These rows do not change the primary epoch-99 comparison. Photo uncertainty resamples paired originals; it does not account for dependence between originals in the same source album.

Photo AUCs are rounded to six decimal places; a displayed 1.000000 can be a rounded value. The JSON retains full precision.

| Checkpoint | Role | Frame AUC | Photo AUC |
|---|---|---|---|
| rgb_ep099 | primary | 0.952 [0.907, 0.961] | 1.000000 [1.000000, 1.000000] |
| gray_ep099 | primary | 0.953 [0.949, 0.965] | 1.000000 [1.000000, 1.000000] |
| p99_ep099 | primary | 0.948 [0.919, 0.956] | 1.000000 [1.000000, 1.000000] |
| rgb_ep073 | secondary | 0.949 [0.871, 0.961] | 1.000000 [1.000000, 1.000000] |
| gray_ep058 | secondary | 0.948 [0.930, 0.955] | 0.999992 [0.999978, 1.000000] |
| p99_ep053 | secondary | 0.950 [0.904, 0.960] | 1.000000 [0.999999, 1.000000] |

## Measurement-agreement subset

This subset contains 1836 frames (368 sharp and 1468 blurred). Labels agree with the independent classical measurement, which filters the teacher-labelled cohort; this does not create independent human ground truth.

| Checkpoint | Subset AUC |
|---|---|
| rgb_ep099 | 0.982 [0.956, 0.993] |
| gray_ep099 | 0.997 [0.991, 1.000] |
| p99_ep099 | 0.996 [0.984, 0.998] |
| rgb_ep073 | 0.982 [0.939, 0.991] |
| gray_ep058 | 0.996 [0.979, 1.000] |
| p99_ep053 | 0.994 [0.976, 0.999] |

## Interpretation and limits

- Only one training run per input and seven test video groups are represented; no training-seed uncertainty is estimated.
- Percentile bootstrap coverage is uncertain with seven imbalanced groups; intervals describe conditional group sampling, not all possible deployment domains.
- A validation recall target is not guaranteed on held-out groups; report realised test recall at the unchanged threshold.
- Teacher agreement and agreement with a classical measurement do not establish objective human-labelled blur accuracy.
- Synthetic photo recipe labels, blur lengths and original-pair uncertainty do not validate separation of native blur types; source-album dependence is not included in the photo intervals.
- The earlier checkpoints were selected using external evaluation/selection sets; this report does not remove that selection dependence.
- Paired difference intervals are descriptive; they are not equivalence tests, and no multiple-comparison adjustment is applied.
- The frozen cohort and deterministic synthetic recipes are evaluated with the recorded software and weights; no claim of immutable historical source pixels is made.

The full thresholds, anonymous group counts, per-group rates, leave-one-group-out results, paired differences and file hashes are in [heldout_results.json](heldout_results.json). Private frame IDs, video names, source paths and raw scores are not distributed.
