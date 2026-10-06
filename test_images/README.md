# Test images

Two groups:
- **`typical/`**: frames on which the teacher and all three models (epoch 99) agree, and one photo with and without
  added blur. They show the normal behaviour. They were chosen by a fixed rule, not by hand: among the demo frames whose
  teacher code is `sss` (or `bbb`) and on which all three models give the same class at the 95% operating point, the
  first four in a hashed order; the photo is the first in the same order among those the three models separate.
- **`hard_cases/`**: four deliberately difficult frames, a stress test. Each shows a boundary of the task definition or a
  known weakness; a "wrong" answer here is the point of the example, not a summary of the models' performance (see the
  README for the measured results).

Scores are the models' blur score (sigmoid output; not a calibrated probability); in brackets the decision at the three
stored operating points r90 / r95 / r98 (`s` = sharp, `b` = blur). The scores were computed in batches with bfloat16 on a GPU; a single image on another device can differ in the
third decimal. The demo uses r95 by default; the decision can depend
on the operating point, as `tos_05_4f_00456` shows. Teacher code: Q1 Q2 Q3 (METHOD §4); `sbb` = no blur seen, but not
sharp.

## typical

| file | teacher code | p99 | grayscale | RGB |
|---|---|---|---|---|
| `tos_04_2b_00304.png` | `sss` | 0.000 (s/s/s) | 0.000 (s/s/s) | 0.000 (s/s/s) |
| `tos_07_3a_00270.png` | `sss` | 0.000 (s/s/s) | 0.001 (s/s/s) | 0.000 (s/s/s) |
| `tos_09_1a_00332.png` | `sss` | 0.000 (s/s/s) | 0.000 (s/s/s) | 0.000 (s/s/s) |
| `tos_03_2c_00542.png` | `sss` | 0.000 (s/s/s) | 0.000 (s/s/s) | 0.000 (s/s/s) |
| `tos_01_1a_01328.png` | `bbb` | 1.000 (b/b/b) | 1.000 (b/b/b) | 1.000 (b/b/b) |
| `tos_07_1b_02007.png` | `bbb` | 0.993 (b/b/b) | 0.927 (b/b/b) | 1.000 (b/b/b) |
| `tos_06_1b_00415.png` | `bbb` | 1.000 (b/b/b) | 1.000 (b/b/b) | 1.000 (b/b/b) |
| `tos_01_2a_01167.png` | `bbb` | 1.000 (b/b/b) | 1.000 (b/b/b) | 1.000 (b/b/b) |
| `uhdiqa_2494_L00.png` (no blur added) | – | 0.001 (s/s/s) | 0.000 (s/s/s) | 0.000 (s/s/s) |
| `uhdiqa_2494_L16.png` (16 px simulated camera blur) | – | 1.000 (b/b/b) | 1.000 (b/b/b) | 1.000 (b/b/b) |

## hard_cases

| file | what it shows | teacher code | p99 | grayscale | RGB |
|---|---|---|---|---|---|
| `tos_01_1a_00164.png` | water surface with reflections | `bbb` | 0.000 (s/s/s) | 0.003 (s/s/b) | 0.000 (s/s/s) |
| `tos_02_4a_00326.png` | soft, not sharp picture; poorly lit, defocused foreground | `sbb` | 0.986 (b/b/b) | 0.000 (s/s/s) | 0.000 (s/s/s) |
| `tos_05_4f_00456.png` | sharp subject, defocused background (intentional depth of field) | `sss` | 0.809 (s/b/b) | 1.000 (b/b/b) | 0.154 (s/b/b) |
| `tos_07_2b_00170.png` | local motion blur of a moving hand in an otherwise sharp shot | `bbb` | 0.000 (s/s/s) | 0.002 (s/s/s) | 0.001 (s/s/s) |


What each hard case tests:
- **water surface**: moving water reads as blur to a local measure and to the teacher; whether such a frame is usable is a
  question of the task definition, not of edge sharpness.
- **soft picture with a defocused foreground**: not usable, but not motion blur either (`sbb`); only the fingerprint
  model flags it, through the defocused foreground (its cell map), not through the soft subject.
- **sharp subject, defocused background**: intentional depth of field. The image decision is "blur if blur appears
  anywhere" (METHOD §8), so a defocused background can turn the decision; whether that is wanted depends on the
  definition of usable.
- **local motion blur**: a real miss of all three models: a small, dark, low-contrast moving region in an otherwise sharp
  shot. Why it is missed (size of the region, contrast, composition, the pooling) is not yet separated.

All images are 1080p and went once through H.264 (CRF 23), the coding chain of METHOD.md (Section 6).

- **Tears of Steel** (`tos_*`): (CC) Blender Foundation | mango.blender.org, licensed under CC BY 3.0
  (https://creativecommons.org/licenses/by/3.0/). Changes: the original 4K OpenEXR camera frames were converted to sRGB
  (exposure 1), centre-cropped to 16:9, reduced to 1920 × 1080 by 2 × 2 averaging in linear light and encoded as one
  H.264 frame.
- **UHD-IQA Benchmark Database** (`uhdiqa_*`, https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html), CC0.
  Changes: resized to a 1080 px short side; the `L16` image has simulated camera motion blur of 16 px (METHOD.md,
  Section 6); both encoded as one H.264 frame.
