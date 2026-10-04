# Method

*Draft. Experiments are still running; numbers marked as preliminary will be updated.*

## 1. Task and scope

The goal is a **universal, blur-specific detector for video frames**: a model that separates sharp frames from frames
degraded by motion blur, independently of the film, the shot or the recording. The first stage, described here, is a
binary decision (sharp / blurred). A later stage will distinguish the type of blur (object motion, camera motion,
defocus).

Sharpness is treated as an **absolute, source-independent** property: the task is not to find the best frames of a
video, but frames that are acceptably sharp whatever their source. Low-quality material is meant to be filtered out,
never taught as acceptable.

The cost of the two errors is **asymmetric**. A blurred frame accepted as sharp is the expensive error; rejecting some
sharp frames is acceptable. All evaluation is built around this (Section 9).

**How the experiment evolved.** The experiment started as training a detector on the blur fingerprint alone (the edge
residual of Section 7), on the hypothesis that the finest edge layer carries the blur pattern while suppressing the
content. Its results, in particular that the fingerprint alone reached a high level, prompted the comparison with the
plain RGB image and with a grayscale image, which separates the contribution of the frequency band (fingerprint vs
grayscale) from that of colour (grayscale vs RGB).

**Prior work this builds on.** The edge fingerprint (Section 7) comes from our own earlier study,
[RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling), which reconstructs a continuous RGB mesh from
pixel-centre samples and examined in detail how much edge and thin-feature energy different mesh reconstructions lose,
and how motion-blur patterns appear in that loss. The blur detector uses that study's simplest reconstruction, `m09`,
as its edge detector.

**Scope of the results.** All statements hold for the material used here: people-centred films and videos with a
narrowed theme, recorded in extreme close-up, close-up, medium and long shots, indoors and outdoors, with backgrounds
and many objects. Testing on other domains (nature films, other genres) is future work.

## 2. Material

| source | role | size |
|---|---|---|
| video frames | real, native motion blur; both classes | 177 imported videos (H.264, 1920×1080, 4:2:0); 37 labelled by the VLM teacher (Section 4); ~4.3 M frames in total, 374,019 of them labelled |
| photos | sharp anchor + simulated camera blur | 48,566 camera photos, de-duplicated. Part of them: only files with camera EXIF data (genuine photographs, no video frames), resized to a 1080 px short side; the rest: photos at native size with a fixed-height border at the bottom, which is cut off first (judged to be camera photographs as well) |
| 4K video frames | sharp anchor + simulated camera blur | 231 published 4K video frames (JPEG), from which the blurred ones were removed by hand; reduced to 1920×1080 by exact 2×2 averaging. They carry a noise / grain floor like the camera photos (and a JPEG 8 px grid), not the clean floor of the encoded 1080p video frames |

**Why these videos.** The frames come from NSFW videos. This material concentrates the cases that are hard for
measurement-based sharpness filtering: large unstructured skin surfaces with little edge texture, water scenes (moving
water reads as blur), extreme close-ups and close-ups, and fast, often periodic motion of several people at once. The
videos were shot with handheld cameras between 2014 and 2026, with various camera models, so recording quality,
lighting and compression vary widely.

All material is protected by copyright and **is not published** (Section 12).

## 3. Frames: extraction and measurement

**Extraction.** Every frame of a video is decoded at its native frame rate (consecutive frames are ~33 ms apart) and
stored losslessly. Frames carrying subtitles or overlay text are excluded.

**Shots.** A "shot" here is not the film-editing notion. It is a run of near-identical frames, cut by a simple frame
similarity: each frame is described by the mean colour of a 16 × 9 grid, and two frames are compared by a trimmed
distance over the best-matching 65% of the cells (so a person moving through the picture does not dominate). A shot
ends at a sudden change between neighbouring frames (cut distance ≥ 0.12) or when the picture has drifted far enough
from the shot's first frame (drift distance ≥ 0.08, e.g. a zoom or a slow reframing); a shot lasts at least 0.5 s.
The segmentation is therefore not consistent with the editor's cuts: a long take may be split into several shots,
and the unit is "interchangeable framing", which is what the per-shot ranking needs. A content filter removes empty or unusable frames: luminance standard deviation ≥ 1, entropy ≥ 1 bit,
at most 90% over- or under-exposed pixels.

**Burnt-in text and watermark.** The selected frames contain no burnt-in text (subtitles, captions, overlays). Every
frame carries a watermark: semi-transparent white, at a fixed position per video, so it is barely or not visible on a
light background. A per-video mask marks its area.
- *Measurement:* the watermark area is excluded, so it does not influence the measured values.
- *Training:* the watermark is masked on half of the frame samples and left visible on the other half (Section 8).
  On the unmasked samples it appears whole or, on a light background, only partially; the model has to learn that this
  sharp overlay does not make a blurred frame sharp. Masked pixels never contribute to the decision (Section 7).

**Measurement** (an independent, non-learned signal, used for ranking and as a cross-check, never as the label):

- *camera motion*: the global displacement between neighbouring frames, estimated per 256 px tile and taking the
  median of the tiles (at least 4 tiles with enough contrast), so a hand crossing the picture cannot drag it;
- *regional motion*: the displacement left after compensating the camera, measured in 160 px cells;
- *within-shot sharpness rank*: the rank of a frame's sharpness among the frames of its shot (smaller = sharper).

**How motion is measured.** A frame difference equals displacement times local contrast, so it is divided by the local
gradient to give pixels. Three pairs of frames are compared around frame *n*:
- the **outer pair** *n−1* / *n+1* (halved, as two frame intervals separate them) is the main signal. At a dead point of
  a fast, periodic movement the motion reverses: *n−1* and *n+1* return to nearly the same place while frame *n* sits at
  the extreme, held still for its exposure. Frame *n* takes no part in this comparison, so it is not misread as moving;
- the two **inner pairs** *n−1* / *n* and *n* / *n+1* are a floor for a turn that lasts longer than one frame (otherwise
  the first frame of such a plateau, with a moving neighbour on one side, would read as fast).

The smallest of the three wins. Neighbours that are exact copies of frame *n* (duplicated source frames) are skipped,
because they would force the reading to zero. Only structured pixels count (the strongest 15% of gradients, per frame
or per cell), the 90th percentile is taken within a cell, and the regional value is the 90th percentile over the cells
(the tenth-worst region: not a single noisy cell, and not the calm majority when the motion is confined to a corner).
Cells covering the watermark or with too little structure do not vote.

The motion values are displacements between frames, not blur lengths: the visible blur depends on the exposure time.
At the usual 180° shutter the smear is about half the measured travel.

### 3.1 Variety of the sample and the handheld camera

The frames span many shots (3,819 labelled shots in 36 videos), subjects, locations, interactions, lighting set-ups,
camera models, operators and years of production (2014–2026), and therefore many compression levels. Within a shot
the variety is small: neighbouring frames are variations of one picture. Between shots and videos it is large.

**The share of sharp frames varies widely.** How many sharp frames a video or a shot can give depends on the recording
quality, the camera motion, the motion of the subjects, the camera view (close-up or long shot), the number of people,
and on water or a moving water surface in the picture; each of these can reduce the number of sharp frames
substantially. Per video (37 labelled videos), the `sss` share of the labelled frames ranges from 0.2% to 77.6%
(median 29.9%), the blurred (`bbb` + `bsb`) share from 3.2% to 60.5% (median 20.9%). Per shot:

| blurred share of the shot | shots with `sss` frames | `sss` frames | `sss` per shot (median) | `sss` share of the shot (median) |
|---|---|---|---|---|
| < 10% | 1,383 | 84,328 | 31 | 65% |
| 10–25% | 372 | 15,624 | 21 | 39% |
| 25–50% | 331 | 7,999 | 9 | 24% |
| 50–75% | 236 | 2,690 | 5 | 15% |
| > 75% | 216 | 1,130 | 2 | 5% |

This is why the sharp selection is proportional per shot (Section 5): a calm shot (static camera, slowly moving subject)
gives many near-identical sharp frames, a blur-dominated shot only a few. Those few are the most valuable borderline cases, because in such a shot they alone
carry the sharp pattern.

The handheld camera matters for two reasons. First, the camera moves continuously, so the native blur is a mixture of
camera blur and object motion blur. Second, the measured camera motion shows what the teacher still accepts as sharp.
Displacement between neighbouring frames (px); the measurement covers ~4.2 M frames, the rows below are the 374,019
labelled ones:

| label | camera, median | camera, 99th pct. | regional (after camera), median | regional, 99th pct. |
|---|---|---|---|---|
| `sss` | 1.9 | 21.6 | 5.5 | 17.9 |
| `ssb` | 1.4 | 23.3 | 5.5 | 18.5 |
| `bsb` | 2.3 | 35.4 | 9.3 | 26.3 |
| `bbb` | 3.4 | 49.3 | 14.4 | 36.6 |

Camera motion is present in sharp frames as well: a frame-to-frame camera displacement of up to ~20 px often leaves no
visible blur, because the visible blur depends on the exposure time. The classes overlap strongly in displacement, which
is why displacement alone cannot be used as a blur length, and why the measurement ranks frames within a shot instead of
labelling them.

## 4. Labelling with a frozen VLM teacher

Every frame is put to a frozen vision-language model (Qwen3.5-4B) three times, with three different questions. The
answers are kept separately as a three-letter code, one letter per question: `s` sharp, `b` blurred, `u` unreadable
(the reply contained no explicit yes / no).

| question | text | a "yes" means |
|---|---|---|
| **Q1** | *Is this image blurry or does it have motion blur? Yes or no?* | blurred |
| **Q2** | *Is this video frame sharp? Yes or no?* | sharp |
| **Q3** | *Is this image sharp? Yes or no?* | sharp |

**Choice of model and prompts.** Several vision-language models and model sizes were tried (Qwen3.5 at 0.8B, 2B, 4B and 9B among them);
Qwen3.5-4B (bf16) was chosen: it was as reliable as the larger model on the test images and faster. Its main advantage for this material is that it can judge blur also on extreme
close-ups and close-ups and on large, unstructured skin surfaces, where edge-based measurements have little to work
with. The prompts were shaped by testing them on many images: simple, short, closed (yes / no) questions whose answers
are easy to parse. Like every such model, it is prompt-sensitive, which was also seen during the processing.

**Design of the three questions.** The questions were built on purpose so that contradictory, uncertain labels can be
recognised from the combination of the answers. Q1 asks whether the image is blurry and explicitly whether it has motion
blur. Q2 and Q3 are control questions about sharpness, one about the *video frame*, one about the *image*; they judge
sharpness in a more complex way, weighing the whole picture, including its composition (for example an intentionally
soft background). With the word "frame" the teacher
answers more leniently, with "image" more strictly. A frame is accepted as sharp only if all three agree (`sss`);
combinations in which the answers contradict each other mark the uncertain cases and are left out.

The wording is the threshold. In a pilot on 156 frames, Q1 called 24% blurred, Q2 32% and Q3 72%: Q3 is a much stricter
version of Q2 that differs by one word.

**Identity of the teacher.** The teacher is a byte-identical frozen copy (code, prompt, generation settings, weights),
identified by hashes. Before a video is labelled, three control images with recorded replies are asked again; any
difference in the reply text stops the run.

**Batched answering.** Answers are obtained in batches of four images with greedy decoding, which forces a bare
yes / no. In a pilot, batched and single-image sampled answering differed on about 1 in 40 frames, at the same speed,
and the batched answers were judged better on the test images. The choice was made for an unambiguous, short,
machine-readable output.

**Label codes used.** `sss` = sharp. `bbb` and `bsb` = blurred (Q1 sees blur). Excluded as uncertain: `bss`, `sbb`,
`ssb`.

### 4.1 The role of human judgement

Manual quality labelling usually follows one of two approaches. Either a single rater labels all samples, which keeps
the criteria and the decision strategy relatively consistent over the whole dataset; or several independent raters
judge, and their decisions are combined by majority, consensus or statistical aggregation, in which case inter-rater
agreement and rater variability can also be quantified.

In this experiment the role of human decisions was deliberately minimised. The perceptual boundary between a sharp and
a slightly blurred image is not a well-defined, directly observable threshold. The human visual system does not judge an
image from local information alone: context, familiar structures, edge and contour continuity and perceptual completion
partly compensate for the degradation, so an image that is already measurably unsharp may still look acceptably sharp.
The judgement is also strongly influenced by the reference and by the relative quality of the images shown one after
another, so it becomes partly comparative.

A further problem of the boundary region is that when the rater no longer sees a clear perceptual difference between the
two categories, the uncertainty of the decision grows. Under a forced binary choice, decisions on such samples become
partly random, which lowers the reproducibility and reliability of the labels.

Human judgement was therefore not used as direct ground truth, but mainly to calibrate the VLM-based labelling: shaping
and testing the prompts on sample images, choosing between batched and single-image answering on the frames where they
differed, and visually inspecting graded blur ladders. A similar transitional uncertainty was also seen in the teacher's
decisions (Section 6.1: the share of `sss` answers does not fall monotonically between 1 and 3 px). So for synthetically
increasing blur no single sharp / blurred dividing line was drawn. Instead, two stable ranges were set: the largest blur
at which an image still counts clearly as sharp (3 px, from the teacher probe, Section 6.1), and the smallest blur at
which it counts clearly as blurred (16 px, from visual inspection of encoded blur ladders: from 16 px on even extreme
close-ups look blurred). The uncertain transition band between the two is excluded from training and from the photo
evaluation.

This lowers the risk that the model learns a categorical decision from samples on which the visual property itself
cannot be determined reliably by a binary human judgement.

## 5. Frame selection

**Label noise.** The teacher's label is never changed by the measurement: the measurement does not relabel a frame and
does not move it to the other class. It is used only to reduce label noise inside a class, in this order:
1. the content filter and the exclusion of frames with burnt-in text (Section 3);
2. the exclusion of uncertain answer combinations (`bss`, `sbb`, `ssb`) and of unreadable answers;
3. on the sharp side only, the per-shot ranking by the measured sharpness, which keeps the best part of the `sss` frames
   of each shot (rules below).

The sharp and blurred sides are selected by different rules, on purpose.

- **Sharp (`sss`).** The teacher gives the label, the measurement ranks. Per shot only the best part of the `sss`
  frames by within-shot sharpness rank is kept: 20% of the shot's `sss` frames, 10% if more than half of the shot is
  blurred, at least one, with at least 0.2 s between kept frames (neighbouring sharp frames are near-duplicates).
  This reduces the teacher's tolerance: an `sss` frame may still hide a small blur the teacher did not notice.
- **Blurred (`bbb`, `bsb`).** Not filtered: every frame is kept, with no minimum gap (neighbouring blurred frames
  differ because the motion changes them). This covers the widest possible range of motion blur: weak and strong,
  partial and full, slow and fast, periodic.

**Splits.** Whole videos go to train, validation or test; duplicate videos share one group. Validation and test are
balanced per video (as many sharp as blurred frames from every video), so a video's look carries no label information.

| split | sharp | blurred | video groups |
|---|---|---|---|
| train | 13,198 | 58,364 | 18 |
| validation | 3,301 | 3,301 | 11 |
| test | 3,296 | 3,296 | 7 |

**Frames as pairs.** Within a shot, sharp and blurred frames share scene, light, camera, codec and people; what differs is
the native motion blur. 1,183 of the 2,502 training shots contain both classes (71% of the sharp and 62% of the blurred
training frames). Consecutive blurred frames are permutations of one motion, 33 ms apart.

## 6. Photos: simulated camera blur through the video path

The photos supply the two **extremes** with exact labels; the frames supply the ambiguous middle.

**Why the video path.** A camera photo's high-frequency content is dominated by sensor noise, a fine point cloud that motion
blur turns into streaks. Video frames do not carry this noise: the video codec removes it.
On an unencoded photo, synthetic camera blur therefore produces a characteristic hatched pattern: the noise is smeared
into fine parallel streaks along the motion direction, visible in the fingerprint even at 3 px and extremely
conspicuous. A model trained on it would learn this hatching instead of the blur of real frames; this is a main reason
why every photo is re-encoded (after the blur) before training. Measured on the flat parts
of the images, the noise floor of the frames is matched only after H.264 encoding (CRF 23 closest); JPEG does not match
(too strong an 8 px block grid). So every photo sample goes through the same chain as a video frame:

> original → flip / 90° rotation → camera motion blur in linear light → H.264 (x264 High, 4:2:0, BT.709, one frame) → decoded

**Labels by blur length L** (px on the photo grid; linear, uniform motion):

| class | L | reason |
|---|---|---|
| sharp | 0, 1, 2, 3 px | tolerance of a sharp *frame* (Section 6.1) |
| not trained | 3–16 px | the ambiguous band is left to the frames |
| blurred | 16, 20, 24, 30 px | from 16 px on, even extreme close-ups look blurred |

**Systematic, not random variants.** Each class has a fixed grid of 4 lengths × 12 directions (every 15°) × 4 CRF values
(20, 23, 26, 28) = 192 points. Every photo walks this grid with a stride co-prime to its size, so each epoch gives every
photo a different variant and the grid points are evenly spread over the pool at every epoch. Samples are made on the fly.

**Blur kernel.** The motion segment is convolved with the linear-interpolation (tent) reconstruction of the image, so the
kernel is correct for small L (L = 1 px already blurs; its spread matches the theoretical value from 1 px on).

**Watermark masks.** 50% of the photo samples receive a watermark mask taken from the frames, so the mask is not a
photo / frame marker.

### 6.1 Calibrating the sharp tolerance with the teacher

60 held-out photos, each at L = 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6 and 16 px (then H.264), were put to the teacher.
The original is sharp by definition; on 7 of 60 originals the teacher did not answer `sss` (6× `ssb`, 1× `bss`), which
is counted as a teacher error. Among the 53 photos whose original was `sss`:

| L (px) | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| still `sss` | 50 | 50 | 49 | 47 | 49 | 48 | 44 | 40 | 31 | 0 |
| Q1 says blurred | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 | – |
| Q2 says not sharp | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 4 | 6 | – |

At 16 px no photo stayed `sss` (over all 60 photos: 56 answered blurred to Q1).

Up to 3 px the teacher answers almost as on the original (the remaining changes come from Q3, at the level of its own
error on the originals). Q2 reacts from 4 px, Q1 notices motion blur from about 6 px. Consequences: declaring 0–3 px
sharp does not contradict the frame labels; and `sss` frames may contain a 3–5 px blur the teacher did not notice.

The tolerable blur depends on the shot: an extreme close-up can take ~8 px without looking blurred, a long shot shows
even 2 px. A single pixel threshold cannot fit both; this is why the photos give only the extremes and the middle band
is learned from the frames.

### 6.2 Two groups calibrated to each other

- **Photos (discrete images):** the sharp side simulates the tolerance (edge thickness, 0–3 px), aligned with the
  teacher but not stricter than it; the blurred side gives clean camera blur, from which the model may in principle also
  learn the smearing of the pattern caused by motion.
- **Frames (native blur):** selected by the teacher's label, then filtered by the measurement (sharp side) to reduce
  the label uncertainty.
- The transition band is removed from both groups: 3–16 px on the photos, the uncertain codes and the weaker `sss`
  frames on the frames.

## 7. Model input

**Edge fingerprint.** The edge detector is our own, taken from the earlier
[RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling) study rather than a standard operator (Sobel,
Laplacian), because that study had examined in detail the edge loss of the different mesh reconstructions and how
blur patterns show in it. `m09` is the 9-node, uncalibrated biquadratic (Q2) mesh built from the pixel-centre samples
(corner and edge-midpoint nodes reconstructed from the neighbouring samples), integrated back over each source pixel's
own area. On the identity grid this equals a fixed 3×3 kernel `K = [3 10 3; 10 92 10; 3 10 3] / 144` with replicated
borders (verified numerically to 2·10⁻¹⁵). The fingerprint is the part of the image this reconstruction does not keep:
`d = Y − K * Y`, where `Y` is the linear-light luminance (Rec. 709 weights). In the mesh study this residual is exactly
the edge and thin-feature contrast that the uncalibrated mesh smooths away and that area calibration restores. `d` keeps only the finest edge layer: a doubled contour or
parallel bands along the motion direction appear as paired positive / negative lines.

**Why the sign matters.** At an edge, `d` is negative on the darker side and positive on the brighter side. Which of the
two lines of a pair is the outer and which the inner one is therefore set by the brightness of the subject against the
background and by the direction of the motion; their spacing follows the blur length and their orientation the motion
direction. The signed pair thus encodes geometry that a gradient magnitude or a spectrum loses, and it is a strong cue.

**What is seen in the fingerprint as blur grows** (visual observation): first the edges thicken. Beyond that the picture
varies from one motion blur to another, depending on how cleanly the edge doubles: a clean displacement between two
positions produces two edges, and the contour lines double regularly. On video frames the patterns span a much wider
range than this clean case (uneven speed, curved paths, partial and repeated motion, coding). It is computed on the GPU, never
stored. Per image it is divided by its 99.5th percentile magnitude (floor 10⁻³) and companded as `sign · √|·|`, so only
the shape of the pattern counts, not the overall sharpness level of the recording.

**RGB.** As an alternative, the plain ImageNet-normalised image.

In both cases a fourth (or second) channel carries the validity mask; masked pixels are set to zero.

### 7.1 Relation to known approaches

*Based on the authors' knowledge; a systematic literature review is still to be done, and references will be added.*

- **Residual-domain inputs in image forensics.** Camera identification and manipulation detection have long fed CNNs
  not the image but a high-pass residual (fixed rich-model filters, or a first layer constrained to be high-pass), to
  suppress the content and expose fine pixel-level traces. The edge fingerprint used here is this idea applied to blur,
  with signed edge pairs (the doubled contour and the parallel bands of motion blur appear as paired positive / negative
  lines, which a gradient magnitude would lose) and per-image normalisation (only the shape of the pattern counts).
- **Gradient magnitude or Fourier spectrum as input.** A Sobel / Laplacian magnitude drops the sign of the edges; a
  global spectrum drops where the blur is. Here the blur is often partial (one moving hand), video coding removes high
  frequencies from sharp frames as well, and noise or coding artefacts add them to blurred ones; so the model decides per
  16 px cell on the signed pattern, and the image is blurred if blur appears anywhere.
- **Blur segmentation and kernel estimation.** Pixel-level blur maps are usually learned from small hand-labelled sets
  and often do not separate defocus from motion blur; patch-wise motion-kernel estimation learns from synthetic straight
  blur. The per-cell output here is a coarse blur map; a finer map would need a decoder.
- **Less common, as far as the authors know:** labels from a frozen VLM teacher on real frames, with label noise reduced
  by an independent measurement and with the two classes selected by different rules; photo pairs passed through the
  video path (blur, then H.264) to match the frames' noise floor, with the sharp tolerance calibrated against the
  teacher; the two groups calibrated to each other (photos give the extremes, frames the middle band); and an asymmetric
  operating-point metric instead of accuracy.

## 8. Model, loss and sampling

**Model.** A ConvNeXt (Small; Base and Large are prepared) trunk up to stride 16, ImageNet weights (a DINOv3 variant was
also run). The first layer is adapted to the input channels; the mask channel starts at zero. A 1×1 head gives one logit
per 16 px cell; the image logit is a masked log-sum-exp over the cells with a learnable sharpness r ("blurred if blur
appears anywhere", differentiable). Cells covered less than half by valid pixels are excluded.

**Loss.** Image-level binary cross-entropy plus a dense term (weight 0.5) on cells that contain edges: on photos every
edge cell carries the image label (a camera motion moves every edge); frames have no dense term (where the blur is in a
frame is unknown).

**Sampling.** Each epoch draws 12,000 samples from four strata: sharp frames, blurred frames, sharp photos and blurred
photos, 3,000 each. Inside a stratum every sample is equally likely and drawn without repetition until the stratum is
exhausted, then reshuffled. There is no weighting by video or source: the selection decides which frames take part, the
sampler does not re-weight them. Flips and 90° rotations are applied per sample (lossless).

| stratum | pool | one full pass | passes in 100 epochs |
|---|---|---|---|
| sharp frames | 13,198 | ~4.4 epochs | ~23 |
| photos (each class) | 33,484 | ~11.2 epochs | ~9, each time a different grid variant |
| blurred frames | 58,364 | ~19.5 epochs | ~5 |

The watermark is masked on 50% of the frame samples and visible on the other 50%: the model has to learn that a sharp
overlay does not make a blurred frame sharp.

### 8.1 Repetition: passes over the data, and frames as permutations of shots

An epoch draws 3,000 samples from each of the four strata, without repetition inside a stratum until it is exhausted,
then reshuffled. The strata differ in size, so they are revisited at different rates:

| stratum | pool (training) | one full pass | passes in 100 epochs | what a repetition is |
|---|---|---|---|---|
| sharp frames (`sss`) | 13,198 | ~4.4 epochs | ~22.7 | the same frame (only flips / 90° rotations differ) |
| photos, each class | 33,484 | ~11.2 epochs | ~9.0 | a different variant every time (192-point grid of length, direction, CRF) |
| blurred frames (`bbb`, `bsb`) | 58,364 | ~19.5 epochs | ~5.1 | the same frame |

**Why the analysis counts passes over the blurred frames.** The blurred frames are the largest pool and the slowest
cycle: only after one blurred-frame pass has the model seen every training sample at least once. Within one such pass
the sharp frames are revisited about 4.4 times and the photos about twice. The validation curves oscillate with these
cycles, so results are compared as means per blurred-frame pass, not as single epochs.

**The frames are discrete samples, but not independent ones.** Counted back to shots (the near-identical framing units of
Section 3):

| | value |
|---|---|
| training shots | 2,502 |
| shots that give blurred frames | 1,943 (median 15, mean 30, max 672 blurred frames per shot) |
| shots that give sharp frames | 1,742 (median 3, mean 7.6, max 152 sharp frames per shot) |
| shots that give both (pairs) | 1,183, holding 62% of the blurred and 71% of the sharp training frames |
| blurred frames per sharp frame in these paired shots | ~3.9 |
| contiguous runs of consecutive blurred frames (≤ 0.05 s apart) | 11,875; median 2 frames, mean 4.9, 90th percentile 11, max 523 |

Consecutive blurred frames 33 ms apart are permutations of one motion; each run is close to one blur situation. So the
58,364 blurred frames come from about 11,900 runs in 1,943 shots, and over 100 epochs each run is drawn about 25 times
on average (in different neighbouring frames). In the 1,183 paired shots each sharp frame stands against about four
blurred frames of the same scene, light, camera and coding, which is the native counterpart of a photo's sharp /
blurred pair. The number of distinct blur situations, not the number of blurred frames, is therefore what bounds what
the model can learn about unseen videos.

## 9. Evaluation

- **Photo level (exact labels):** AUC on held-out photos, 2,000 sharp and 2,000 blurred, one fixed variant each.
- **Frame level (teacher labels):** AUC, and the **share of sharp frames kept at a threshold that catches 90%, 95% or 98% of
  the blurred frames**. The 98% figure is the strict operating point of the asymmetric cost.
- These figures measure **agreement with the teacher**, not strictness: the teacher's `sss` frames include overlooked
  blur, which a stricter model rejects rightly.
- **Confident subsets:** frames whose teacher label is confirmed by the measurement (sharp: `sss`, rank ≤ 0.2, regional
  motion ≤ 4 px, camera ≤ 1.5 px; blurred: `bbb`/`bsb`, rank ≥ 0.8, regional motion ≥ 10 px). 533 sharp and 1,788 blurred
  validation frames. Reported: AUC on them, and the confident-sharp share kept when at most 1% of the confident-blurred
  frames pass.
- **Train side:** every model was evaluated after every epoch not only on unseen data but also on a fixed subset of its
  own training data (~6,600 training frames, balanced per video, plus 800 training photos, the same subset in every
  epoch), with the threshold taken from the validation frames. This shows how well the model learned the training
  images themselves and how the gap to unseen videos develops, which is not possible in every training setup.

  Sharp training frames kept at 95% blurred recall (train / validation, the gap, and the frame AUC on both sides),
  mean per pass over the blurred frames, the three publishable 100-epoch runs:

  | pass | edge fingerprint (p99) | grayscale | RGB |
  |---|---|---|---|
  | 1 | 71.1 / 60.5% (gap 10.6), AUC 0.930 / 0.914 | 71.5 / 62.7% (8.8), 0.943 / 0.925 | 80.5 / 69.3% (11.2), 0.957 / 0.941 |
  | 2 | 84.0 / 68.4% (15.5), 0.965 / 0.935 | 84.1 / 70.9% (13.2), 0.972 / 0.940 | 87.7 / 74.5% (13.2), 0.980 / 0.948 |
  | 3 | 89.8 / 72.8% (17.0), 0.978 / 0.944 | 87.9 / 73.4% (14.5), 0.982 / 0.945 | 88.5 / 72.6% (15.9), 0.987 / 0.948 |
  | 4 | 91.8 / 73.6% (18.2), 0.987 / 0.947 | 88.9 / 71.6% (17.3), 0.989 / 0.944 | 92.5 / 76.1% (16.4), 0.993 / 0.953 |
  | 5 | 93.6 / 73.6% (20.0), 0.992 / 0.949 | 91.2 / 71.6% (19.6), 0.994 / 0.944 | 94.1 / 75.1% (18.9), 0.997 / 0.953 |

  Observations: on the training frames every input keeps improving to the end (AUC 0.992–0.997, 91–94% kept), while on
  unseen videos the curves flatten after the third pass; the gap therefore grows steadily (to 19–20 points at 95%).
  None of the models reaches 100% on its own training frames. RGB learns its training images fastest; the edge
  fingerprint starts lowest but reaches the highest training level of the two colourless inputs; grayscale has the
  smallest gap early on but the lowest training level at the end.
- The validation curve oscillates with the data cycles (Section 8); single epochs are not compared, but averages over
  blur passes.

## 10. Training environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| OS | Windows 11 Pro |
| software | Python 3.13, PyTorch 2.14 with CUDA 13.0, x264 via FFmpeg |
| precision | bfloat16 autocast, gradient checkpointing |
| optimiser | AdamW, weight decay 0.05, gradient clipping 1.0 |
| learning rate | trunk 1·10⁻⁴, head 1·10⁻³; warm-up 0.5 epoch; single cosine to 1% |
| batch | 8 images × 2 accumulation steps (effective 16), batches bucketed by image shape |
| epoch | 12,000 samples, ~13.6 min including validation (~25 images/s) |
| runs | no early stopping; every epoch saved |
| labelling (VLM teacher) | Qwen3.5-4B, bf16, on the same GPU; three questions per image, batches of 4, greedy decoding |
| labelling throughput | ~2,900 images/h in a standalone run on 1080p photos (no other load on the machine) |
| labelling memory | up to ~28.7 GB of GPU memory at 1620 × 1080 inputs in batches of 4 (largely the allocator's cache), utilisation fluctuating ~30–93% (image loading and answer parsing run on the CPU between batches) |

**Provenance.** Each publishable run records, at start, the SHA-256 of every source file, the configuration, the index,
the frame selection and the initial weights, plus the software and hardware versions; a resume with changed code is
refused.

## 11. Experiment history

| run | change | lesson |
|---|---|---|
| first round | frames + photos with simulated camera blur | a selection that produced blur-only videos let the model recognise the video instead of the blur (frame AUC ~0.5) |
| balanced round | both classes from every video | frame AUC 0.94 after 12 epochs; ~80 / 70 / 55% sharp kept at 90 / 95 / 98% |
| photos only | no frames in training | photos solved at once; frames AUC 0.80–0.84: photo blur alone does not transfer |
| photos + blurred frames only | sharp side photos only | collapse: every frame called blurred ("frame = blurred" via the watermark mask and video look) → frames must be in both classes |
| H.264 photo pairs + frames | photos through the video path | frame AUC up to 0.945 |
| on-the-fly photos (final recipe) | systematic grid, one photo pool, no source / video weighting, masks 50 / 50 | frame AUC up to 0.946 (60-epoch schedule, ran 30 epochs) |
| DINOv3 ConvNeXt-Small | other pre-training, same recipe | behind ImageNet in every phase at this learning rate; ~3× noisier epoch to epoch; ends at AUC 0.936 |
| **RGB, 100 epochs** (publishable) | plain image instead of the fingerprint | per-pass means settle at AUC 0.953, ~85 / 75–76 / 62–63% |
| **edge fingerprint, 100 epochs** (publishable) | same recipe, fingerprint input | slower start; reaches the RGB level by the 3rd pass (AUC 0.944, 82.9 / 72.8 / 57.5%); ends at AUC 0.949, ~83 / 73.6 / 58% |
| **grayscale, 100 epochs** (publishable) | image without colour (linear luminance) | at the level of the other two by the 3rd pass (AUC 0.945, 84.5 / 73.4 / 55.2%); no further gain, ends at AUC 0.944, 84.0 / 71.6 / 54.6% |

Preliminary reading: all three inputs converge to a similar level by the third pass over the blurred frames. RGB learns
faster and more evenly and ends a few points ahead (about 2 points at 95%, 4–5 points at 98% in the last two passes); the
edge fingerprint alone, a lossy imprint from which the image cannot be reconstructed, already carries enough information to
recognise the blur. The training results show that the information is present in all three inputs, not which property
carries the blur or whether the models use the same one. This is the subject of a planned cross-test of every model on
RGB, grayscale and edge-only images (two renderings), at several compression levels and with sensor noise, and of an
ablation with masked edges.

**Note on resumed runs.** Each of the three publishable runs was resumed once from a checkpoint (RGB after an operating-
system restart; edge fingerprint and grayscale after the same data-loader error at the same step). On resume the sampler order of the interrupted epoch is reshuffled, so a resumed run is not
sample-for-sample identical to an uninterrupted one. The data-loader error was traced to a single synthetic frame whose
raw bytes happened to begin with "ID3", which the video encoder's input parser took for a metadata tag. The photo synthesis
has since been fixed (such a frame is passed in BGR byte order) and covered by a regression test; the three runs
were made before the fix.

## 12. Data availability

The dataset is **not published**: it was built from copyrighted videos and photo albums. No images, frames, crops or
derived images (edge images, heat maps) are published. Code, configurations, selection and labelling rules, run
provenance, metrics and trained weights are. Demonstration images will come from a publishable source.
