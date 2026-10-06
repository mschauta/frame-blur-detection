# Method

## 1. Task and scope

The goal is a **universal, blur-specific detector for video frames**: a model that separates sharp frames from frames
degraded by blur, motion blur in particular, independently of the film, the shot or the recording. The first stage,
described here, is a binary decision (sharp / blurred). A later stage will distinguish the type of blur (object motion,
camera motion, defocus).

**Blur types are not separated in this experiment.** On the frames of a video the blur phenomena rarely appear in a
pure form: camera motion blur, the motion blur of a moving subject and defocus usually occur together, and which of them
causes the impression of unsharpness would have had to be analysed frame by frame. The labels reflect this. Q1 identifies
the phenomenon relatively well (blur, motion blur), but it does not answer whether the frame is sharp. Q2 and Q3 ask
about sharpness; their answers are influenced by motion blur, but the teacher also recognises other types of blur with
them (Section 4). Separating motion blur alone would need further experiments with the prompts. The synthetic blur of the
photos (Section 6), on the other hand, deliberately produces camera motion-blur patterns (uniform, straight motion); this
pattern rarely occurs in a pure form in films, if only because it would make them unwatchable.

**The purpose is a quality classification.** The detector is meant to find the frames that can be used, not merely to
filter out blurred ones: its decision serves a keep / discard classification of frames for a dataset. A frame is *usable*
if it is sharp in the sense of this study; a blurred frame is not, and neither is a frame whose detail is destroyed by
other degradations, such as heavy lossy compression with visible artefacts. The model's two classes are named *sharp*
and *blurred*; in use, *blurred* stands for "not usable".

Sharpness is treated as an **absolute, source-independent** property: the task is not to find the best frames of a
video, but frames that are acceptably sharp whatever their source. Low-quality material is meant to be filtered out,
never taught as acceptable.

The cost of the two errors is **asymmetric**. A blurred frame accepted as sharp is the expensive error; rejecting some
sharp frames is acceptable. All evaluation is built around this (Section 9).

**Not a distillation of the teacher's decisions.** The class labels of the real frames do come from the VLM teacher, but
the detector is not trained to reproduce its sharp / blurred decisions: this is a filtered weak supervision with
synthetic anchors. The teacher's labels only help to sort the frames into the two classes: only `sss` (all three questions say
sharp) and `bbb` / `bsb` (Q1 sees blur) are used, the uncertain and the not-sharp-but-not-blurred combinations are left out, an independent measurement reduces the label
noise on the sharp side, and the two extremes are defined without the teacher, by photos with an exact injected blur length (an
unblurred photo is sharp even where the teacher says otherwise). Agreement with the teacher is therefore reported as
agreement, not as accuracy (Section 9).

**How the experiment evolved.** The experiment started as training a detector on the blur fingerprint alone (the edge
residual of Section 7), on the hypothesis that the finest edge layer carries the blur pattern while suppressing the
content. Its results, in particular that the fingerprint alone reached a high level, prompted the comparison with the
plain RGB image and with a grayscale image: grayscale vs RGB differ in colour; fingerprint vs grayscale mainly in the
frequency band, but also in the encoding (linear residual vs sRGB-encoded luminance), the normalisation, the companding
and the parametrisation of the first layer.

**Prior work this builds on.** The edge fingerprint (Section 7) comes from our own earlier study,
[RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling), which reconstructs a continuous RGB mesh from
pixel-centre samples and examined in detail how much edge and thin-feature energy different mesh reconstructions lose,
and how motion-blur patterns appear in that loss. The blur detector uses that study's simplest reconstruction, `m09`,
as its edge detector.

**Scope of the results.** All statements hold for the material used here: people-centred films and videos with a
narrowed theme, recorded in extreme close-up, close-up, medium and long shots, indoors and outdoors, with backgrounds
and many objects. Testing on other domains (nature films, other genres) is future work.

### 1.1 Sharp and blurred: two ends of a scale

*Sharp* and *blurred* are not exact categories but the two ends of a scale. Where a picture stops being sharp is partly a
question of composition and aesthetics:
- How much displacement and how much loss of sharpness is tolerated?
- How disturbing is the blur of one object in the composition?
- Is the blur an error, or part of the picture (a panned background, a deliberately soft foreground)?

Blur also has many types: motion blur from the camera or from the subject, focus and defocus, and surfaces that are
soft or hazy in themselves (mist, water, smooth skin, an out-of-focus plane). The question "is this frame sharp?" sounds
simple; answering it is much harder. This was confirmed both by the measurements (Section 3.1: the classes overlap
strongly in measured displacement) and by the vision-language models (Section 4.2: two usable model sizes disagree on a
substantial share of the same frames). The extremes are easy to identify, also for human perception; the transitions are
uncertain whether one relies on a model, on a measurement or on a human observer.

**Human perception.** How much blur the eye tolerates and when it notices it depends on the viewing conditions, not only
on the image:
- *Resolution of the eye.* Normal visual acuity resolves detail of about one minute of arc. Whether a blur of a few
  pixels is visible therefore depends on how large the picture is seen: the display size and the viewing distance
  (pixels per degree), not the pixel count alone. A frame viewed small can look sharp; the same frame enlarged does not.
- *Contrast and content.* Blur is noticed first on strong, high-contrast edges and fine regular structure; on smooth,
  untextured areas (skin, sky, water) the same blur can stay invisible because there is little fine detail to lose.
- *Context and completion.* The visual system judges a picture as a whole: familiar shapes, edge and contour continuity
  and perceptual completion partly compensate for the degradation, so a measurably unsharp image may still look sharp
  (Section 4.1).
- *Reference and adaptation.* The judgement depends on what was seen before: after looking at blurred pictures, a
  slightly soft picture looks sharper, and the order of the pictures shown makes the judgement partly comparative.
- *Motion.* In a moving picture, blurred frames often go unnoticed: in playback the eye integrates over time and
  follows moving subjects, and a moving blurred pattern looks sharper than the same pattern standing still. Motion blur
  that is invisible at playback becomes visible when the single frame is looked at. This matters here, because the frames
  are meant to be used as single images.

**A working figure from the frame measurement, and why it was not enough.** The exposure time of the videos is not
known; 35 of the 37 labelled videos run at 29.97 frames per second (about 33 ms between frames), two at 59.94. Assuming the usual 180° shutter, the
exposure is about half the frame interval, so the smear is about half the measured travel between neighbouring frames
(Section 3). As a working figure, a smear of two to three pixels on a 1080p frame was taken as the point where the eye
begins to see it. Because the visibility depends on how near the camera is, thresholds per shot size were then tried
(a travel of 1 px for long and medium shots, 3 px for medium close-ups, 5 px for close-ups and extreme close-ups). This
needed the shot size of every shot, which no model determined without errors (Section 4.2). The working figure is an
assumption, not a measurement; the sharp tolerance of the photos (0–3 px, Section 6.1) was later set independently, with
the teacher, and is consistent with it.

These observations are why no single pixel threshold is used to define the classes (Section 4.1, Section 6.1): the
photos give the two ends of the scale with an exact blur length, and the uncertain middle is learned from the frames.

*References for the perceptual statements will be added; a systematic literature review is still to be done.*

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

**Real and synthetic blur.** Real blur exists only on unmanipulated video frames: the labelled frames of the dataset
and the test frames (Section 9). Every other blur in this study is synthetic: the camera blur added to the photos
(Section 6), the blur ladders of the test photos, and, among the reference sets, blur made by averaging consecutive
frames. The original photos themselves, both the training photos and the test photos, may contain blur as part of the
composition (a defocused background, a soft foreground, a deliberately blurred subject). In this study *sharp* for an
original photo means "no blur was added", not "no blur anywhere in the picture": in training, such compositional blur
is therefore taught as sharp (Section 6), and in testing it has to be kept in mind when a model calls an original
blurred.

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

**Measurement** (an independent, non-learned signal, used for ranking and as a cross-check, never as the label).
The measured values, including single-image sharpness measures such as the Laplacian variance, were used only to
measure, select and filter the frames. They were never used in training: they are neither labels nor model input. The
model sees only the RGB-mesh `m09` edge fingerprint with p99 normalisation (Section 7), or, in the comparison runs, the
RGB or grayscale image. The measurement consists of:

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

Every frame is put to a frozen vision-language model (Qwen3.5-4B-bf16) three times, with three different questions. The
answers are kept separately as a three-letter code, one letter per question, in the order Q1, Q2, Q3; `u` marks an
unreadable reply (no explicit yes / no). **The letters do not mean the same for every question:**
- **Q1** asks about blur and motion blur: `b` = blurry / motion blur, `s` = *no* blur seen. An `s` on Q1 does **not**
  mean that the image is sharp, only that Q1 saw no blur.
- **Q2 and Q3** ask explicitly about sharpness: `s` = sharp, `b` = *not* sharp.

So `sbb` is not a sharp code: Q1 sees no blur, but both sharpness questions say the image is not sharp (for example a
soft, low-detail or defocused picture). Only `sss`, where all three questions agree, counts as sharp.

| question | text | a "yes" means |
|---|---|---|
| **Q1** | *Is this image blurry or does it have motion blur? Yes or no?* | blurred |
| **Q2** | *Is this video frame sharp? Yes or no?* | sharp |
| **Q3** | *Is this image sharp? Yes or no?* | sharp |

**Choice of model and prompts.** Several vision-language models and model sizes were tried (Qwen3.5 at 0.8B, 2B, 4B and 9B among them);
Qwen3.5-4B-bf16 was chosen as the teacher. The four sizes were later compared under identical conditions (Section 4.2). Its main advantage for this material is that it can judge blur also on extreme
close-ups and close-ups and on large, unstructured skin surfaces, where edge-based measurements have little to work
with. The prompts were first tested interactively in ComfyUI with several models and weights, on sharp and
blurred images, and in particular on extreme close-ups and close-ups that the measurement rated blurred but that looked
sharp to the eye. In these tests the teacher resolved that contradiction: its answers followed human perception rather
than the measurement. The experiments with the two sharpness questions were prompted by the observation that on some
images the teacher judged too strictly with the Q3 wording and more tolerantly with the Q2 wording. The prompts were then
shaped further by testing them on many images: simple, short, closed (yes / no) questions whose answers
are easy to parse. Like every such model, it is prompt-sensitive, which was also seen during the processing.
It is also content-sensitive. The labels were checked by hand on many videos: on the same image the answers are
reproducible, but how well the teacher follows the prompt depends on the picture content. On some content it follows
the questions well, on other content it does not; for example, whole shots that are sharp to the eye received the
code `sbb` on every frame (no blur seen, but not sharp). Such frames are left out of both classes, so this kind of
error reduces the sharp material rather than putting sharp frames into the blurred class. For these reasons it is hard
to make firm general statements about the teacher. It handles large, untextured skin surfaces well, which is the main
reason it was judged suitable for labelling this material. It also did better on the sharp / blurred questions, which
are answered with yes or no, than on classifying the shot size, where it had to choose one of several views. The
checks also showed that although the question looks simple, the decision behind it is much more complex, much as it
is for a human observer (Section 4.1).

**Design of the three questions.** The questions were built on purpose so that contradictory, uncertain labels can be
recognised from the combination of the answers. Q1 asks whether the image is blurry and explicitly whether it has motion
blur. Q2 and Q3 are control questions about sharpness, one about the *video frame*, one about the *image*; they judge
sharpness in a more complex way, weighing the whole picture, including its composition (for example an intentionally
soft background). With the word "frame" the teacher
answers more leniently, with "image" more strictly. A frame is accepted as sharp only if all three agree (`sss`);
the other combinations are either contradictory, borderline, or not sharp without being motion-blurred, and are left
out (see the code table below).

The wording is the threshold. In a pilot on 156 frames, Q1 called 24% blurred, Q2 32% and Q3 72%: Q3 is a much stricter
version of Q2 that differs by one word.

**Consistency of labels and measurements.** Every frame used here was labelled with the same frozen teacher and
measured with the same method; the earlier labelling and measurements were carried out again, so the labels and measured values of all
frames are directly comparable.

**Identity of the teacher.** The teacher is a byte-identical frozen copy (code, prompt, generation settings, weights),
identified by hashes. Before a video is labelled, three control images with recorded replies are asked again; any
difference in the reply text stops the run.

**Batched answering.** Answers are obtained in batches of four images with greedy decoding, which forces a bare
yes / no. Whether batched answering is acceptable was a judgement made by the author on the test images of a pilot, where
batched and single-image (sampled) answering differed on about 1 in 40 frames; the decision was to label in batches,
for an unambiguous, short, machine-readable output. The teacher is prompt-sensitive and, even more, composition-dependent,
so the pilot figure holds for the pilot images only, and the re-measurement in Section 4.2 for its own samples. Neither
is a general statement; both are observations.

**Label codes used.**

| code | Q1 (blur?) | Q2 (frame sharp?) | Q3 (image sharp?) | use |
|---|---|---|---|---|
| `sss` | no blur | sharp | sharp | **sharp** |
| `bbb` | blur | not sharp | not sharp | **blurred** |
| `bsb` | blur | sharp | not sharp | **blurred** (Q1 sees blur, the strict Q3 agrees) |
| `ssb` | no blur | sharp | not sharp | excluded: borderline (only the strict Q3 objects) |
| `sbb` | no blur | not sharp | not sharp | excluded: not sharp, but no blur seen (not the blurred class either) |
| `bss` | blur | sharp | sharp | excluded: contradictory |

The selection follows this exactly: sharp frames are `sss` only, blurred frames `bbb` and `bsb` only (Section 5).

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

### 4.2 Choosing the teacher: four model sizes under identical conditions

The teacher was first chosen on a small set of test images. The choice was then re-measured in a controlled experiment:
the labeller's own client and wrapper, the same three questions, the same generation settings and the same answer
parser; only the weights changed (Qwen3.5 at 0.8B, 2B, 4B and 9B, all bf16). The approved 4B was verified against its
frozen weight manifest and its control images were replayed; for the other sizes the SHA-256 of every weight file was
recorded. Every model ran in its own process, and the GPU memory was released between models (back to 1.4–2.3 GB
before the next one was loaded).

**Test images.**
- *Photos with an exact injected blur length:* all 1,000 openly licensed demo images from the
  [UHD-IQA Benchmark Database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) (Hosu et al., 2024; CC0;
  a mix of photographs, edited images and renders, not people-centred). They are not training images: none of the
  detectors of this study has seen them. Each was given simulated camera blur of L = 0, 1, 2, 3, 4, 6 and 16 px through the
  video path of Section 6: 7,000 images. L ≤ 3 px counts as sharp, 16 px as blurred; 4 and 6 px are reported only.
- *Frames:* 1,278 confident-sharp frames (up to three per shot, at least 0.5 s apart) and 2,239 confident-blurred
  frames (one per shot) from the labelled videos, with the bounds of Section 9. Their reference is the 4B's own code,
  with which the measurement agrees, so the frames favour the 4B by construction: for the other sizes they show the
  deviation from the 4B, not correctness.
- *Shot size:* no model sized the shots without errors (pose and segmentation models merge or split people who are
  entangled; the VLM mixes neighbouring sizes). The frames are therefore split into two groups only, *close* (close-up
  or extreme close-up) and *not close*, by asking the 4B "*Is this image a close-up or an extreme close-up? Yes or no?*"
  on three frames spread over each shot. A shot is assigned only if all three answers agree: 2,014 close and 542 not
  close shots; 129 shots with disagreeing answers are left out of the split.

**Photos: share of answers per blur length** (`sss` = sharp; blurred = `bbb` or `bsb`; the rest are the other codes of the table in Section 4, mostly `sbb`, `bss` and `ssb`):

| L (px) | 0 | 1 | 2 | 3 | 4 | 6 | 16 |
|---|---|---|---|---|---|---|---|
| 4B `sss` | 78.4% | 77.2% | 76.4% | 73.9% | 71.4% | 63.9% | 5.0% |
| 4B blurred | 5.4% | 5.5% | 7.2% | 7.5% | 8.4% | 11.5% | 78.4% |
| 9B `sss` | 91.9% | 91.2% | 89.9% | 88.3% | 86.3% | 77.7% | 6.6% |
| 9B blurred | 2.3% | 2.7% | 2.4% | 2.7% | 3.3% | 6.0% | 79.3% |
| 2B `sss` / blurred | 30.8% / 1.9% | | | 27.8% / 2.2% | | 20.0% / 4.6% | 0.2% / 58.4% |
| 0.8B `sss` / blurred | 0.0% / 0.4% | | | 0.0% / 0.4% | | 0.0% / 1.1% | 0.0% / 19.3% |

**Frames: same class as the 4B** (in brackets: the opposite class; the rest are the other, excluded codes):

| | sharp, all | blurred, all | sharp, close | blurred, close | sharp, not close | blurred, not close |
|---|---|---|---|---|---|---|
| 4B | 100% | 100% | 100% | 100% | 100% | 100% |
| 9B | 99.1% (0.2%) | 72.2% (15.9%) | 98.4% (0.2%) | 76.0% (13.4%) | 99.3% (0.1%) | 50.6% (30.6%) |
| 2B | 0.3% | 45.5% | 0.0% | 49.8% | 0.5% | 20.4% |
| 0.8B | 0.0% | 14.4% | 0.0% | 15.7% | 0.0% | 4.7% |
| frames | 1,278 | 2,239 | 438 | 1,882 | 761 | 255 |

**Speed and memory** (batches of four, three questions per image; frames 1920 × 1080, photos at their prepared size):

| model | images / h, frames | images / h, photos | GPU memory peak, frames | GPU memory peak, photos |
|---|---|---|---|---|
| 0.8B | 3,916 | 2,723 | 6.5 GB | 16.6 GB |
| 2B | 4,259 | 4,986 | 9.4 GB | 19.7 GB |
| 4B | 2,747 | 2,839 | 16.8 GB | 31.8 GB |
| 9B | 2,045 | 1,974 | 25.1 GB | 32.1 GB |

The memory peak is the device total during the run and includes the allocator's cache.

**Findings.**
- *The two small sizes do not discriminate.* The 0.8B answers "no" to all three questions (not blurry, but not sharp
  either: code `sbb` on 90.8% of the frames and 96.5% of the photos); the 2B answers "yes" to all three (`bss` on
  70.9% and 65.6%). Neither is usable as a blur labeller.
- *The 4B is reproducible.* Run again in batches, it returned exactly the stored code on all 3,517 frames.
- *4B and 9B both follow the blur length:* at 16 px only 5–7% remain `sss`, and they call 78.4% and 79.3% blurred.
  The 9B is more lenient on the sharp side: it calls 91.9% of the unblurred photos `sss`, the 4B 78.4%. By the
  definition of Section 6 an unblurred photo is sharp, so on these images the 4B errs more often on the sharp side.
- *On frames the two differ mainly on the blurred side.* The 9B calls 15.9% of the blurred frames on which the measurement agrees
  sharp, 13.4% in close shots and 30.6% in shots that are not close. Which of the two is right cannot be decided on these
  frames: there is no exact reference, the measurement only agrees on the extremes. This disagreement is the label noise
  the detector is trained through.
- *Close shots are where the VLM matters.* The measurement-based filter does not work reliably on close-ups and extreme
  close-ups; the two usable sizes agree more there (76.0% of the blurred frames) than on shots that are not close (50.6%).
- *Cost.* The 9B is about 25–30% slower than the 4B and needs nearly the whole 32 GB card in batches of four.

*Note.* The model is prompt-sensitive and strongly composition-dependent, and its reasoning mode ("thinking") was
switched off in every run. The figures above hold for these samples only; they are observations, not a general ranking
of the models. They do not claim that the model cannot decide precisely whether a single image is sharp or blurred;
they show that in mass labelling, with short closed questions and no reasoning, label noise has to be taken into
account.

**Batched and single-image answering of the 4B** (observation on these samples; single-image answering samples one image
per request, batched answering decodes four greedily):

| | same code as batched | opposite class (sharp ↔ blurred) | images / h | GPU memory peak |
|---|---|---|---|---|
| frames (3,517) | 90.2% (1 in 10 differs) | 6 frames (0.17%) | 934 (batched: 2,747) | 12.4 GB (batched: 16.8 GB) |
| photos (7,000) | 91.9% (1 in 12 differs) | none | 1,012 (batched: 2,839) | 17.9 GB (batched: 31.8 GB) |

Single-image answering kept `sss` on 78.1% of the unblurred photos (batched: 78.4%) and called 76.6% of the 16 px photos
blurred (batched: 78.4%). Almost every difference between the two modes is a move between a used code and an excluded
one, not between the two classes. On these samples the two modes differ on about 1 in 10 images, more often than in the
pilot (about 1 in 40, Section 4); single-image answering was about three times slower.

The labels of this study are the 4B's, answered in batches.

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

The photos supply the two **extremes** with an exact injected blur length; the frames supply the ambiguous middle.

**What the photos stand for, and why they are blurred.** An unblurred photo (L = 0) is the ideotype of an absolutely
sharp frame; 1–3 px is its realistic abstraction, the tolerance of a sharp frame (Section 6.1). Both are sharp by
definition, and the teacher's label does not override this. The photos are blurred for two reasons. First, every sharp
image gets an absolute blurred counterpart that the model can tell apart from it without doubt, so it sees the sharp and
the blurred pattern in context, on the same content. Second, the synthetic blur produces blur patterns that do not occur
on the video frames, or occur only rarely (uniform straight camera motion in twelve directions and four lengths).

**Why the video path.** A camera photo's high-frequency content is dominated by sensor noise, a fine point cloud that motion
blur turns into streaks. Video frames do not carry this noise: the video codec removes it.
On an unencoded photo, synthetic camera blur therefore produces a characteristic hatched pattern: the noise is smeared
into fine parallel streaks along the motion direction, visible in the fingerprint even at 3 px and extremely
conspicuous. A model trained on it would learn this hatching instead of the blur of real frames; this is a main reason
why every photo is re-encoded (after the blur) before training. Measured on the flat parts
of the images, the noise floor of the frames is matched only after H.264 encoding (CRF 23 closest); JPEG does not match
(too strong an 8 px block grid). So every photo sample goes through the same coding chain as a video frame, with one restriction: a single encoded frame
is intra-coded, so the temporal artefacts of the P / B frames of a real group of pictures are not reproduced; the chain
matches the noise floor, not every artefact of a video:

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

**Why the sign matters.** At an edge, `d` is negative on the darker side and positive on the brighter side. The sign of
a line therefore shows the polarity of the edge (which side is brighter); the spacing of a pair follows the blur length,
and its axis the orientation of the motion. The temporal direction of the motion (from A to B or from B to A) cannot be
told: a uniform exposure kernel is symmetric, so reversing the path gives the same blur. The signed pair still encodes
geometry that a gradient magnitude or a spectrum loses, and it is a strong cue.

**Numerical precision.** In training (and in all evaluations here) the model ran under bfloat16 autocast on the GPU, and
the 3 × 3 convolution of the fingerprint ran in bfloat16 with it: its input, kernel and output were rounded to bfloat16,
while `Y` stayed float32. In smooth regions this rounding is of the order of the fingerprint itself (on a test frame the
median error is 7·10⁻⁵ against a median |d| of 2·10⁻⁴; a perfectly flat image gives inputs up to ±0.74 instead of 0). The
published models were trained on this rounded fingerprint, and the published inference code reproduces the rounding
explicitly on every device. A model trained on the exact float32 fingerprint is future work; its results may differ.

**What is seen in the fingerprint as blur grows** (visual observation): first the edges thicken. Beyond that the picture
varies from one motion blur to another, depending on how cleanly the edge doubles: a clean displacement between two
positions produces two edges, and the contour lines double regularly. On video frames the patterns span a much wider
range than this clean case (uneven speed, curved paths, partial and repeated motion, coding). It is computed on the GPU, never
stored. Per image it is divided by its 99.5th percentile magnitude (taken over the valid pixels of a fixed 2 × 2 sub-grid; floor 10⁻³) and companded as `sign · √|·|`, so only
the shape of the pattern counts, not the overall sharpness level of the recording.

**RGB.** The plain image, ImageNet-normalised; the pretrained first layer is kept unchanged.

**Grayscale.** The image without colour: the same linear-light luminance `Y` the fingerprint is made from (full frequency
band), sRGB-encoded, copied to three channels so that the pretrained first layer is kept, and ImageNet-normalised.
Grayscale vs RGB differ only in colour. Fingerprint vs grayscale differ mainly in the frequency band, but also in the
encoding, the normalisation, the companding and the first layer, so that comparison is not a pure frequency-band control.

In every case a last channel carries the validity mask; masked pixels are set to zero. For the fingerprint the first
layer is adapted: its single fingerprint channel starts from the sum of the pretrained RGB filters.

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
  16 px cell on the signed pattern, and a clearly blurred region can decide the image (soft maximum over the cells).
- **Blur segmentation and kernel estimation.** Pixel-level blur maps are usually learned from small hand-labelled sets
  and often do not separate defocus from motion blur; patch-wise motion-kernel estimation learns from synthetic straight
  blur. The per-cell output here is a coarse blur map; a finer map would need a decoder.
- **Less common, as far as the authors know:** labels from a frozen VLM teacher on real frames, with label noise reduced
  by an independent measurement and with the two classes selected by different rules; photo pairs passed through the
  video path (blur, then H.264) to match the frames' noise floor, with the sharp tolerance calibrated against the
  teacher; the two groups calibrated to each other (photos give the extremes, frames the middle band); and an asymmetric
  operating-point metric instead of accuracy.

## 8. Model, loss and sampling

**Model.** A ConvNeXt (Small; Base and Large are prepared) trunk up to stride 16, ImageNet weights (chosen over a DINOv3
pre-training, Section 11). The first layer is adapted to the input channels; the mask channel starts at zero. A 1×1 head gives one logit
per 16 px cell; the image logit is a masked log-sum-exp over the cells with a learnable sharpness r, normalised by the
number of cells: a differentiable soft maximum. It was chosen so that blur anywhere can raise the image score, but it is
not a logical "any cell" rule: with one dominant cell the image logit is about `local − log(n) / r`, so the size of a
blurred region matters. The cell grid is the output sampling (16 px), not a validated 16 px receptive field or a
segmentation. Cells covered less than half by valid pixels are excluded.

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

- **Photo level (exact injected blur length):** AUC on held-out photos, 2,000 sharp and 2,000 blurred, one fixed variant each.
- **Frame level (teacher labels):** AUC, and the **share of sharp frames kept at a threshold that catches 90%, 95% or 98% of
  the blurred frames**. The 98% figure is the strict operating point of the asymmetric cost.
- These figures measure **agreement with the teacher**, not strictness: the teacher's `sss` frames include overlooked
  blur, which a stricter model rejects rightly.
- **A disagreement with the teacher is not by itself an error of the detector.** It can be a justified stricter (or more
  tolerant) decision of a detector that has learned different principles, an error of the teacher, or an error of the
  detector; without ground truth, which of these it is can only be decided case by case.
  The teacher is a vision-language model that analyses the picture and weighs its composition; the ConvNeXt detector
  learns patterns, and being pretrained on natural images, it is more at home with an RGB or grayscale picture than with
  the edge fingerprint. The teacher's answers (Q1–Q3) act on the dataset only indirectly, through the strongly filtered
  selection of the training frames (Sections 4–5). In the tests the teacher's label is a reference, not ground truth:
  it gives a direction and points to contradictions. The disagreements are then looked at by eye, as a screening test,
  to see which side is closer to reality.
- **Measurement-agreement subsets** ("confident" frames below): frames whose teacher label agrees with a second,
  correlated criterion, the measurement; the measurement does not prove the label (sharp: `sss`, rank ≤ 0.2, regional
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
- **Cross-test.** Every model is also given the finished input of the other two models, its own input processing
  bypassed and nothing converted (channels only copied or selected to fit the first layer), on the same test material
  and at its own operating points. This separates what a model has learned from what its input contains. Results and
  three-set diagrams: README, "Cross-test".
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
| labelling (VLM teacher) | Qwen3.5-4B-bf16, on the same GPU; three questions per image, batches of 4, greedy decoding |
| labelling throughput | ~2,900 images/h in a standalone run on 1080p photos (no other load on the machine) |
| labelling memory | up to ~28.7 GB of GPU memory at 1620 × 1080 inputs in batches of 4 (largely the allocator's cache), utilisation fluctuating ~30–93% (image loading and answer parsing run on the CPU between batches) |

**Provenance.** Each publishable run records, at start, the SHA-256 of every source file, the configuration, the index,
the frame selection and the initial weights, plus the software and hardware versions; a resume with changed code is
refused.

## 11. The publishable runs

| run | input | result (means per pass over the blurred frames, validation) |
|---|---|---|
| RGB, 100 epochs | the plain image | per-pass means settle at AUC 0.953, ~85 / 75–76 / 62–63% sharp kept at 90 / 95 / 98% recall |
| edge fingerprint, 100 epochs | the m09 residual, p99-normalised | slower start; at the RGB level by the 3rd pass (AUC 0.944, 82.9 / 72.8 / 57.5%); ends at AUC 0.949, ~83 / 73.6 / 58% |
| grayscale, 100 epochs | the image without colour (linear luminance) | at the level of the other two by the 3rd pass (AUC 0.945, 84.5 / 73.4 / 55.2%); ends at AUC 0.944, 84.0 / 71.6 / 54.6% |

All three inputs reach a similar level by the third pass over the blurred frames. The training results show that the
information needed to recognise the blur is present in all three inputs; they do not show which property carries it or
whether the models use the same one. This is examined by the cross-test, in which every model is given the finished
input of the other models (Section 9).

**Choice of pre-training.** Before the publishable runs, a ConvNeXt-Small with DINOv3 pre-training was trained with the
same recipe on the edge fingerprint (60 epochs). At this learning rate it stayed behind the ImageNet-pretrained model in
every phase and was about three times noisier from epoch to epoch (final frame AUC 0.936). This comparison decided the
model: the ImageNet-pretrained ConvNeXt was kept for the publishable runs.

**How identical the three publishable runs are, and what the interruptions changed.**

*Configuration.* The three runs share everything except the input (RGB, grayscale, edge fingerprint): the same seed,
index, frame selection, sampling, augmentation, schedule (100 epochs × 12,000 samples, batches of 8 with two
accumulation steps) and initial ImageNet weights.

*Code.* The RGB and the edge-fingerprint runs ran with the same code state (identical SHA-256 digest of every source
file, at the start and at the resume). The grayscale run used a later state that differs in two points only: the
grayscale input mode itself, and a retry of the video-encoder call in the photo synthesis (three attempts, the error
output logged), which changes nothing when the call succeeds. None of the three contains the later "ID3" fix described
below; its alternative byte order was never used in their training.

*Interruptions.* Each run was interrupted once and resumed from its last saved state (saved every 100 optimiser steps);
the code at the resume was verified to be identical to the code at the start.

| run | cause | resumed at | lost and redone |
|---|---|---|---|
| RGB | operating-system restart | epoch 33, batch 256 of 750 (optimiser step 24,700) | the steps after the last save (at least 40) |
| edge fingerprint | data-loader error | epoch 70, batch 378 (step 52,300) | the steps after the last save |
| grayscale | the same data-loader error | epoch 70, batch 378 (step 52,300) | the steps after the last save |

The error stopped the edge-fingerprint and the grayscale runs at the same step because, with the same seed, they draw the
same samples in the same order; in the grayscale run the new retry repeated the failing call three times with the same
result.

*What a resume restores and what it does not.* Restored: the model and optimiser state, the step counter and with it the
learning-rate schedule, and the random-number states of PyTorch and NumPy. Not restored: the sampler's position inside
its draws without replacement, i.e. the shuffled order of each stratum and how far it had got. The resumed sampler starts
every stratum with a fresh shuffle (still derived from the seed and the epoch). Consequences:
- the remaining batches of the interrupted epoch, and all later epochs, contain other samples in another order than an
  uninterrupted run would;
- in each stratum the pass that was interrupted is cut short: the samples it had already drawn are drawn again in the
  new pass, so they receive one more exposure than the rest (at most one partial pass per stratum over the whole run);
- unchanged are the rules that depend only on the seed, the epoch and the sample: the shape grouping of the batches,
  the flips and rotations, the photo variant (blur length, direction, CRF) and the watermark-mask choice; only which
  sample meets which draw changes.

The edge-fingerprint and the grayscale runs were interrupted at the same point and resumed with the same seed, so they
saw the same sequence of samples over the whole run; the RGB run follows the same sequence up to epoch 33, batch 256,
and a different one afterwards. Even without interruptions the runs would not be bit-for-bit reproducible: the
convolution algorithms are auto-tuned on the GPU and bfloat16 arithmetic is not deterministic.

*The data-loader error.* One synthetic photo sample (a 16 px blurred variant, rotated by 90°, CRF 23, in epoch 70) had
raw pixel bytes that happened to begin with the three characters "ID3". The video encoder's input parser takes such a
beginning for an ID3 metadata tag even when the input is declared as raw video, skips it, receives a frame that is too
short, and fails. After the resume the shuffle was different and the error did not recur. The photo synthesis has since
been fixed: when the RGB bytes begin with "ID3", the same image is passed in BGR byte order (identical pixels, so the
same encoded result), which cannot begin with "ID3" as well; a regression test covers it. The three runs were made
before this fix.

## 12. Data availability

The dataset is **not published**: it was built from copyrighted videos and photo albums. No images, frames, crops or
derived images (edge images, heat maps) are published. Published are the method, metrics, run provenance, the inference
code and trained weights; the training, labelling, selection and evaluation code and the configurations will be published
after they have been cleaned of internal references. The demonstration images of the repository come from openly licensed sources (CC BY 3.0 and CC0); their attribution and
the changes made are listed next to them.
