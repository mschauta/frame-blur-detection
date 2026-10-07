# README posters

The [English poster](../../figures/frame_blur_detection_poster_en.png) and
[Hungarian poster](../../figures/frame_blur_detection_poster_hu.png) introduce the experiment with illustrative scenes
and schematic diagrams. Their style follows the author's
[RGB Mesh Resampling poster](https://github.com/mschauta/rgb-mesh-resampling/blob/main/docs/images/english.png).
The scenes were generated for these illustrations; they are not private training images or recorded model predictions.

The posters use the primary epoch-99 results from the [seven-group internal test](../heldout/RESULTS.md).
The numeric table contains native-frame AUC, sharp retention and realised blurred recall at each model's unchanged
validation r95 threshold. The 4,000 photo recipes are a separate test component, not extra native frames in that table.
The sharp/blur scenes explain the composition-aware training target, including acceptable intentional background
defocus. They do not establish validated subject/background segmentation or the features actually learned.

The reconstruction kernel is `[3 10 3; 10 92 10; 3 10 3] / 144`; the residual is illustrated as `d = Y - K * Y`.
BF16 rounding and per-image percentile normalisation are part of the published input implementation (METHOD §7).
The validity-mask illustration shows excluded image regions, not a person segmentation. Symbolic `p_blur` output
cards avoid inventing a measured score. All illustrated model stages are schematic.

The PNG artwork was produced with the built-in image-generation tool and inspected for text, numeric and diagram
accuracy. The [English SVG summary](../../figures/frame_blur_detection_infographic.svg),
[Hungarian SVG summary](../../figures/hungarian.svg) and [codec figure](../../figures/why_video_coding.svg) remain
available and can be regenerated with `reproduction/render_figures.py`.
