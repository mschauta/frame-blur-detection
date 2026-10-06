# Frame Blur Detection

*Draft. The experiments are still running; numbers will be updated. The inference code and weights are included; the
training code will follow.*

The goal is a **universal, blur-specific detector for video frames**: a model that separates sharp frames from frames
degraded by blur, motion blur in particular, independently of the film, the shot or the recording. The results so far
come from a narrow, people-centred material (37 labelled videos) and its validation split; universality is the aim, not
yet a demonstrated property. It is the first stage of a longer pipeline
(selecting sharp frames → recognising the type of blur → restoring blurred frames).

The full description of the material, the labelling, the training and the evaluation is in [METHOD.md](METHOD.md).
This page summarises the approach and the results.

## Principles

- **Sharpness is absolute and source-independent.** The task is not to find the best frames of a video, but frames that
  are acceptably sharp whatever their source. Low-quality material is filtered out, never taught as acceptable.
- **The cost is asymmetric.** A blurred frame accepted as sharp is the expensive error. The main figure is therefore the
  share of sharp frames that survive a threshold strict enough to catch 95% or 98% of the blurred frames.
- **Our definition, not the teacher's.** Frames are labelled by a frozen vision-language model, but the definition of
  sharp is ours: an original photograph is always sharp, and where the teacher disagrees, the teacher is wrong. The
  detector is not a distillation of the teacher's sharp / blurred decisions; the teacher's labels only help to sort
  the frames into the two classes (METHOD §1, §4).
- **Minimal human judgement.** People shaped and tested the prompts and inspected blur ladders; they did not label the
  data (METHOD §4.1).

## Approach in brief

| part | what | details |
|---|---|---|
| frames | real, native motion blur from handheld-camera videos; labelled by a frozen VLM teacher (Qwen3.5-4B-bf16) with three yes / no questions, only unanimous answers kept | METHOD §3–5 |
| label noise | an independent motion and sharpness measurement ranks frames within a shot and keeps only the best sharp frames; it never relabels a frame and is never a model input | METHOD §3, §5 |
| photos | an exact injected blur length at the two extremes: 0–3 px camera blur = sharp, 16–30 px = blurred, the band between is not trained; every photo goes through the video path (blur in linear light → H.264 → decode) to match the noise floor of real frames | METHOD §6 |
| inputs | three runs on the same recipe: the RGB image, a grayscale image, and an edge fingerprint (the residual of a 3 × 3 RGB-mesh reconstruction kernel, signed, per-image normalised) | METHOD §7 |
| model | ImageNet ConvNeXt-Small to stride 16, one logit per 16 px cell, masked log-sum-exp pooling ("blurred if blur appears anywhere") | METHOD §8 |
| training | 100 epochs, no early stopping, every epoch saved, provenance recorded (hashes of code, configuration, selection and weights) | METHOD §8, §10 |

## Results (validation, 100 epochs)

Validation uses **videos never seen in training**, balanced per video (3,301 sharp and 3,301 blurred frames from 11
video groups). The frame labels come from the teacher, so these figures measure **agreement with the teacher**, not
correctness (METHOD §9). The *confident* subset contains only frames whose teacher label agrees with the
measurement (533 sharp, 1,788 blurred).

![Validation metrics per epoch](figures/validation_curves.svg)

*Thin lines: single epochs. Thick lines: mean per pass over the blurred training frames (one pass ≈ 19.5 epochs). The
curves oscillate with the data cycles, so runs are compared by pass means, not by single epochs.*

Mean per pass over the blurred frames: frame AUC / sharp kept at 95% / sharp kept at 98% blurred recall / AUC on the
confident frames.

| pass | RGB | grayscale | edge fingerprint |
|---|---|---|---|
| 1 | 0.941 / 69.3% / 51.4% / 0.990 | 0.925 / 62.7% / 41.5% / 0.990 | 0.914 / 60.5% / 43.0% / 0.985 |
| 2 | 0.948 / 74.5% / 59.7% / 0.989 | 0.940 / 70.9% / 52.9% / 0.991 | 0.935 / 68.4% / 53.1% / 0.986 |
| 3 | 0.948 / 72.6% / 57.7% / 0.990 | 0.945 / 73.4% / 55.2% / 0.992 | 0.944 / 72.8% / 57.5% / 0.987 |
| 4 | 0.953 / 76.1% / 63.2% / 0.991 | 0.944 / 71.6% / 54.0% / 0.992 | 0.947 / 73.6% / 58.2% / 0.988 |
| 5 | 0.953 / 75.1% / 62.0% / 0.992 | 0.944 / 71.6% / 54.6% / 0.993 | 0.949 / 73.6% / 58.4% / 0.990 |

Last epoch (epoch 99):

| input | frame AUC | sharp kept at 90% / 95% / 98% recall | confident AUC |
|---|---|---|---|
| RGB | 0.952 | 84.8% / 74.9% / 61.7% | 0.992 |
| grayscale | 0.942 | 83.1% / 70.0% / 52.1% | 0.993 |
| edge fingerprint | 0.949 | 83.1% / 73.9% / 58.2% | 0.990 |

On the held-out photos (exact injected blur length, 2,000 sharp and 2,000 blurred) every run reaches an AUC of at least 0.999 at every
epoch: the two extremes are solved, the difficulty lies entirely in the real frames.

**Reading.**
- All three inputs converge to a similar level by the third pass over the blurred frames. RGB learns faster and ends a few
  points ahead at the strict operating points (about 2 points at 95%, 4–5 points at 98% in the last two passes).
- On the confident frames, where the teacher label agrees with the measurement, the order changes: grayscale is
  highest from the second pass on, and in the last pass all three lie within 0.004 of each other. Part of RGB's lead at 98% may therefore
  be agreement with the teacher, which also sees the RGB image.
- The edge fingerprint, a lossy imprint from which the image cannot be reconstructed, alone carries enough information
  to recognise the blur. The training results show that the information is present in all three inputs, not which
  property carries it or whether the models use the same one; this is the subject of the cross-test below.

### Training frames against unseen videos

Every run was also evaluated after every epoch on a fixed subset of its own training data, with the threshold taken from
the validation frames (METHOD §9).

![Sharp frames kept at 95% blurred recall, training frames and unseen videos](figures/train_vs_validation.svg)

On the training frames every input keeps improving to the end (frame AUC 0.993–0.997, 91–94% kept at 95%), while on
unseen videos the curves flatten after the third pass; the gap grows to 19–20 points. None of the models reaches 100% on
its own training frames.

## Tests on material outside the training data

Every model is evaluated at its own operating points: the thresholds that catch 90, 95 or 98% of the blurred frames of
its validation split. Nothing is tuned on the test material.

| test set | what it is | reference |
|---|---|---|
| demo photos | 1,000 photos of the [UHD-IQA Benchmark Database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) (CC0), each with simulated camera blur of L = 0, 0.5 … 6 and 16 px through the coding chain of METHOD §6 | exact injected blur length: 0–3 px sharp, 16 px blurred (4–6 px not counted) |
| demo frames | 1,000 raw camera frames of *Tears of Steel* ((CC) Blender Foundation, mango.blender.org, CC BY 3.0), 1080p, one H.264 frame | the teacher's code: `sss` sharp, `bbb` / `bsb` blurred, other codes left out. A reference, not ground truth |
| GoPro pairs | 1,029 sharp / blurred pairs of the GoPro deblurring dataset (Nah et al., CVPR 2017), native and through one H.264 frame (CRF 23) | the pair; the blur is synthetic (see below) |

**How the GoPro blur was made, and why it is new to the detectors.** According to the original paper (Nah, Kim and Lee,
CVPR 2017), the GoPro images were recorded at 240 frames per second; a blurred image is the average of 7 to 13
consecutive frames after linearising the gamma, and its sharp counterpart is the middle frame among those averaged. Such a blur is a sum of
discrete copies: on one pair (a car's tail light) we counted 7 edge copies about 6.3 px apart. The detectors never saw
blur made this way: the synthetic blur of their training photos is a continuous motion of 0–3 or 16–30 px (METHOD §6),
and the blur of their training frames is native. The GoPro pairs therefore test a blur type outside the training data.

**How to read the three-set diagrams.** Each panel shows the images of one reference class at one operating point. The
three circles hold the images that the RGB, grayscale and edge-fingerprint model (epoch 99) call *blurred*; the numbers
are image counts per region, and the number outside the circles counts the images all three call *sharp*. In the
upper row (reference sharp) everything inside the circles is a false alarm; in the lower row (reference blurred)
everything inside is caught and the number outside is missed by all three models. The diagrams show not only how many
errors a model makes, but whether the models make the same ones.

![Three-set diagram, demo photos](figures/venn/venn_ladder_ep099.svg)

*Demo photos (exact injected blur length). At every operating point all 1,000 photos with 16 px blur are caught by all
three models. On the sharp side (7,000 images, 0–3 px) the false alarms are mostly different for each model; only a small
part is common to all three.*

![Three-set diagram, demo frames](figures/venn/venn_demo_frames_ep099.svg)

*Demo frames, measured against the teacher's code (agreement, not accuracy). A large common core of caught frames, and a
group of frames the teacher calls blurred that all three models call sharp; the decisions of the edge-fingerprint model
are almost entirely contained in those of the other two.*

![Three-set diagram, GoPro native](figures/venn/venn_gopro_native_ep099.svg)

*GoPro pairs, native.*

![Three-set diagram, GoPro through H.264](figures/venn/venn_gopro_h264_ep099.svg)

*GoPro pairs through H.264. After coding, weak blur is missed more often (by all three models); the RGB model catches the
largest number of blurred images that the other two miss.*

The same diagrams for the first epochs (0–5), when the models are still close to their pretrained weights, are in
[figures/venn](figures/venn). Results of the cross-test, in which every model is given the finished input of the
other models, will be added.

## Demo: sort images into sharp / blur

```
pip install -r requirements.txt
python classify.py --weights weights/p99_ep099.pt --input test_images/typical --output sorted
```

Every image of `--input` is scored and copied to `sorted/sharp` or `sorted/blur`; `sorted/results.csv` lists the blur
score and the decision (the sigmoid output of the model; a score, not a calibrated probability). `--recall 90 | 95 | 98` selects the decision threshold stored with the weights (the one
that catches that share of the blurred validation frames; higher = stricter), `--move` moves instead of copying. A GPU is
used when available. The models were trained for 1080p video frames decoded from H.264 and stored as lossless PNG.
In use, *blur* stands for "not usable" (METHOD §1). `test_images/typical` shows the normal behaviour,
`test_images/hard_cases` a deliberate stress test; their scores at all three operating points are listed in
`test_images/README.md`.

## Weights

| file | input | epoch | thresholds r90 / r95 / r98 | validation frame AUC |
|---|---|---|---|---|
| `weights/rgb_ep099.pt` | RGB image | 99 (last) | 0.885 / 0.0200 / 0.0041 | 0.952 |
| `weights/gray_ep099.pt` | grayscale image | 99 (last) | 0.281 / 0.0135 / 0.0028 | 0.942 |
| `weights/p99_ep099.pt` | edge fingerprint | 99 (last) | 0.889 / 0.0323 / 0.0053 | 0.949 |

ConvNeXt-Small (ImageNet-pretrained) to stride 16, about 34 M parameters; each file holds the state dict and its metadata
(input type, epoch, thresholds). The last epoch is the main result; further checkpoints selected from the cross-test
will be added. The files are stored with Git LFS (`git lfs install` before cloning). The weights were trained on
non-public material (METHOD §12).

## Data availability

The training data is **not published**: it was built from copyrighted videos and photographs. No images, frames, crops or
derived images are published. The method, configurations, selection and labelling rules, run provenance, metrics and
trained weights are (METHOD §12).

## Related work by the author

- [RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling): the continuous RGB-mesh reconstruction from
  which the edge fingerprint is derived.

## License

[MIT](LICENSE) © Dr. Schauta Marcell
