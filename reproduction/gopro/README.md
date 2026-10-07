# GoPro mirror provenance

The evaluation copy was downloaded from the user-confirmed
[rahulbhalley/gopro-deblur Kaggle mirror](https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur), version 1.
It contains **2,058 PNG files: 1,029 blurred images and 1,029 corresponding sharp images**, rather than 2,058 pairs.
The evaluation results in README apply to this copy.

[gopro_provenance.json](gopro_provenance.json) records the public download identity, archive size and SHA-256.
[gopro_member_manifest.json](gopro_member_manifest.json) records the public mirror member names, byte lengths,
CRC32 and per-file SHA-256, without publishing the images or local paths. The existing extracted files matched all
archive central-directory sizes and CRC32 values; sharp/blur filenames formed 1,029 complete pairs. These checks read
file bytes without decoding images, running a model or downloading another copy.

The downloaded archive is 2,046,286,654 bytes and has SHA-256
`d43f49edd39314de280b80d5da42d499063e8edb28dadfaa1e3945797be28588`.
The original download metadata also identifies Kaggle dataset 627736, version ID 1118216. Signed URL query parameters
are excluded from this record.

## What is still unknown

The [original GoPro paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)
reports 1,111 test pairs. The mirror's flattened filenames have not been matched to the original sequences/frame IDs;
the missing 82-pair correspondence is therefore not established. The
[official dataset page](https://seungjunnah.github.io/Datasets/gopro.html) also supplies gamma-corrected and linear-CRF
blur variants. The mirror-to-original variant correspondence remains unverified. Do not report these results as a
complete official 1,111-pair test evaluation or infer which variant was used from the original paper alone.

## Verify the same local copy

Python standard library only; no GPU, network, extraction or image decoding:

```text
python reproduction/gopro/verify_gopro_archive.py --archive private/archive.zip --extracted private/gopro_deblur --out work/gopro-verification
```

The extracted directory must contain `blur/images` and `sharp/images`. The script verifies the archive and member
manifest against the recorded copy, then writes only public file identities and aggregate provenance to the chosen
output directory. A mismatch is rejected. This rechecks file identity; it does not map the mirror to the original split.
