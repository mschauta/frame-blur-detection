# Frame Blur Detection

The goal is a **universal, blur-specific detector for video frames**: a model that separates sharp frames from frames
degraded by blur, motion blur in particular, independently of the film, the shot or the recording. The results so far
come from a narrow, people-centred material (37 labelled videos) and its validation split; universality is the aim, not
yet a demonstrated property. It is the first stage of a longer pipeline
(selecting sharp frames → recognising the type of blur → restoring blurred frames).

![Frame Blur Detection infographic](figures/frame_blur_detection_infographic.png)

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
[figures/venn](figures/venn). ### Cross-test: every model on the other models' input

Each model is given the finished input of the other models, its own input processing bypassed and nothing converted
(channels are only copied or selected to fit the first layer; the edge-fingerprint model, whose first layer takes one
image channel, gets the grayscale input as is and the RGB input one colour channel at a time). Same test material,
same operating points. Epoch 99, operating point 95%:

| model ← input | demo photos AUC | sharp photos kept | 16 px caught | demo frames AUC (vs teacher) | GoPro AUC native / H.264 |
|---|---|---|---|---|---|
| RGB ← own | 0.999 | 89.9% | 99.9% | 0.810 | 0.968 / 0.945 |
| RGB ← grayscale | 1.000 | 69.4% | 100% | 0.899 | 0.980 / 0.976 |
| RGB ← fingerprint | 0.741 | 97.4% | 8.0% | 0.575 | 0.534 / 0.840 |
| grayscale ← own | 1.000 | 89.1% | 100% | 0.718 | 0.978 / 0.929 |
| grayscale ← RGB | 1.000 | 90.7% | 100% | 0.692 | 0.977 / 0.922 |
| grayscale ← fingerprint | 0.732 | 48.3% | 81.3% | 0.798 | 0.555 / 0.845 |
| fingerprint ← own | 0.999 | 95.2% | 100% | 0.675 | 0.931 / 0.912 |
| fingerprint ← grayscale | 0.699 | 0.9% | 99.9% | 0.576 | 0.763 / 0.767 |
| fingerprint ← R / G / B channel | 0.65–0.72 | 0.5–1.4% | 99.8–100% | 0.46–0.71 | 0.74–0.75 |

Observations:
- **RGB and grayscale are interchangeable.** The grayscale model decides on the RGB input practically as on its own;
  the RGB model also sees the blur in the grayscale image (it keeps fewer sharp photos, but separates the demo frames
  and the GoPro pairs even better). Colour does not carry decisive information; the two image models rely on features
  present in both.
- **No transfer between the fingerprint and the image, in either direction.** The image models call the fingerprint
  almost always sharp; the fingerprint model calls the full image almost always blurred. The two families have learned
  different features: the image models rely on something the residual does not contain, and the fingerprint model on
  something only the residual contains.
- **Early epochs differ.** After the first epoch the fingerprint model still decides well on the grayscale image (demo
  photos AUC 0.931, GoPro native 0.940), while its first layer and trunk are still close to the ImageNet weights; by
  epoch 5 this is gone. The fingerprint model moves to the residual early in training.
- Consistent with this, the fingerprint model depends less on the general look of an image: on native and H.264 GoPro
  images it keeps 97% and 99.8% of the sharp ones, whereas the image models shift native images towards blurred during
  training (RGB: 98.7% kept after epoch 0, 64.2% after epoch 99). The price is that a blur made of sharp copies (the
  GoPro averaging) misleads it more: it catches 73% of the native GoPro blur, the RGB model 97%.

In the three-set diagrams below all three models are given the same input; the circles are the models.

<details><summary>Given the RGB image (fingerprint model: green channel): three-set diagrams (epoch 99)</summary>

![Demo photos, cross-test, Given the RGB image (fingerprint model: green channel)](figures/venn_crosstest/venn_ladder_ep099_given_rgb.svg)

![Demo frames, cross-test, Given the RGB image (fingerprint model: green channel)](figures/venn_crosstest/venn_demo_frames_ep099_given_rgb.svg)

![GoPro, native, cross-test, Given the RGB image (fingerprint model: green channel)](figures/venn_crosstest/venn_gopro_native_ep099_given_rgb.svg)

![GoPro, H.264, cross-test, Given the RGB image (fingerprint model: green channel)](figures/venn_crosstest/venn_gopro_h264_ep099_given_rgb.svg)

</details>

<details><summary>Given the grayscale image: three-set diagrams (epoch 99)</summary>

![Demo photos, cross-test, Given the grayscale image](figures/venn_crosstest/venn_ladder_ep099_given_gray.svg)

![Demo frames, cross-test, Given the grayscale image](figures/venn_crosstest/venn_demo_frames_ep099_given_gray.svg)

![GoPro, native, cross-test, Given the grayscale image](figures/venn_crosstest/venn_gopro_native_ep099_given_gray.svg)

![GoPro, H.264, cross-test, Given the grayscale image](figures/venn_crosstest/venn_gopro_h264_ep099_given_gray.svg)

</details>

<details><summary>Given the edge fingerprint: three-set diagrams (epoch 99)</summary>

![Demo photos, cross-test, Given the edge fingerprint](figures/venn_crosstest/venn_ladder_ep099_given_p99.svg)

![Demo frames, cross-test, Given the edge fingerprint](figures/venn_crosstest/venn_demo_frames_ep099_given_p99.svg)

![GoPro, native, cross-test, Given the edge fingerprint](figures/venn_crosstest/venn_gopro_native_ep099_given_p99.svg)

![GoPro, H.264, cross-test, Given the edge fingerprint](figures/venn_crosstest/venn_gopro_h264_ep099_given_p99.svg)

</details>

The diagrams of the first epochs (0–5) are in [figures/venn_crosstest](figures/venn_crosstest).

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
