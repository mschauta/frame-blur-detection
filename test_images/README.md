# Test images

| file | source | what it shows | teacher code (Q1 Q2 Q3) |
|---|---|---|---|
| `tos_01_1a_00164.png` | Tears of Steel raw footage | camera frame | `bbb` |
| `tos_05_4f_00456.png` | Tears of Steel raw footage | sharp subject, defocused background (a trap for "blur anywhere") | `sss` |
| `tos_07_2b_00170.png` | Tears of Steel raw footage | motion blur of a moving hand in an otherwise sharp shot | `bbb` |
| `tos_02_4a_00326.png` | Tears of Steel raw footage | soft, not sharp picture with a defocused foreground | `sbb` |
| `uhdiqa_1005_L00.png` | UHD-IQA | photo, no blur added | – |
| `uhdiqa_1005_L16.png` | UHD-IQA | the same photo with 16 px simulated camera blur | – |

All images are 1080p and went once through H.264 (CRF 23), the video path of METHOD.md (Section 6).

- **Tears of Steel:** (CC) Blender Foundation | mango.blender.org, licensed under CC BY 3.0
  (https://creativecommons.org/licenses/by/3.0/). Changes: the original 4K OpenEXR camera frames were converted to
  sRGB (exposure 1), centre-cropped to 16:9, reduced to 1920 × 1080 by 2 × 2 averaging in linear light and encoded as one
  H.264 frame.
- **UHD-IQA Benchmark Database** (https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html), CC0. Changes: resized
  to a 1080 px short side; the second image has simulated camera motion blur of 16 px (METHOD.md, Section 6); both
  encoded as one H.264 frame.
