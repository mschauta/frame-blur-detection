"""Verify an existing GoPro mirror archive against extracted files, without decoding images.

Reads local bytes only. No download, inference, extraction or original-source matching.
The manifest contains public mirror member names and checksums, never local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
import zlib
from pathlib import Path, PurePosixPath


def file_hash(path: Path):
    checksum = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
            crc = zlib.crc32(block, crc)
    return checksum.hexdigest(), f"{crc & 0xffffffff:08x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--extracted", required=True, type=Path,
                        help="Directory containing blur/images and sharp/images")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    known = json.loads((Path(__file__).parent / "gopro_provenance.json").read_text(encoding="utf-8"))
    if args.archive.stat().st_size != known["archive_bytes"]:
        raise ValueError("Archive length differs from the recorded mirror copy")
    archive_sha, _ = file_hash(args.archive)
    if archive_sha != known["archive_sha256"]:
        raise ValueError("Archive SHA-256 differs from the recorded mirror copy")
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        for item in sorted(archive.infolist(), key=lambda row: row.filename):
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            if (len(parts) != 4 or parts[0] != "gopro_deblur" or
                    parts[1] not in {"blur", "sharp"} or parts[2] != "images" or
                    not parts[3].endswith(".png") or ".." in parts):
                raise ValueError("Unexpected archive member layout")
            local = args.extracted.joinpath(*parts[1:])
            if local.stat().st_size != item.file_size:
                raise ValueError("An extracted file differs in length")
            sha, crc = file_hash(local)
            expected = f"{item.CRC:08x}"
            if crc != expected:
                raise ValueError("An extracted file differs from the archive CRC32")
            rows.append({"archive_member": item.filename, "bytes": item.file_size,
                         "crc32": crc, "sha256": sha})
    names = {kind: {row["archive_member"].split("/")[-1] for row in rows
                    if row["archive_member"].split("/")[1] == kind}
             for kind in ("blur", "sharp")}
    if names["blur"] != names["sharp"]:
        raise ValueError("Mirror sharp/blur member names do not form complete pairs")
    for kind in ("blur", "sharp"):
        local_names = {path.name for path in (args.extracted / kind / "images").glob("*.png")}
        if local_names != names[kind]:
            raise ValueError("Local PNG inventory differs from the mirror archive")
    manifest_bytes = (json.dumps(rows, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(manifest_bytes).hexdigest() != known["manifest_sha256"]:
        raise ValueError("Member SHA-256 manifest differs from the recorded mirror copy")
    provenance = {
        "mirror_url": "https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur",
        "mirror_title": "gopro_deblur",
        "kaggle_dataset_id": 627736,
        "kaggle_version_number": 1,
        "kaggle_version_id_from_download_metadata": 1118216,
        "archive_host_path": "https://storage.googleapis.com/kaggle-data-sets/627736/1118216/bundle/archive.zip",
        "download_date_utc": "2026-10-04",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_sha,
        "pair_count": len(names["sharp"]),
        "png_count": len(rows),
        "archive_member_checks": "All extracted PNG sizes and CRC32 values match the existing archive's central directory; SHA256 is recorded for each local member.",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verified_without_image_decoding": True,
        "source_evidence": [
            "User explicitly confirmed this Kaggle mirror URL.",
            "Extracted PNG download metadata references the existing local archive.",
            "Archive download metadata identifies Kaggle dataset 627736/version 1118216; signed query parameters are omitted.",
            "Kaggle public dataset metadata associates dataset 627736 with rahulbhalley/gopro-deblur, version 1, and this archive byte size."
        ],
        "original_dataset_page": "https://seungjunnah.github.io/Datasets/gopro.html",
        "original_paper_url": "https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf",
        "original_paper_test_pairs": 1111,
        "original_mapping_status": "Not established: the flattened mirror names have not been mapped to original GoPro sequences/frame IDs or verified against the official 1111-pair test split.",
        "blur_variant_status": "Not established: the official dataset provides both gamma-corrected and linear-CRF blur variants; the mirror-to-original variant mapping is unverified.",
        "claims_supported": "Results apply to this verified 1029-pair mirror copy, not automatically to the complete official GoPro test benchmark."
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gopro_member_manifest.json").write_bytes(manifest_bytes)
    (args.out / "gopro_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(names["sharp"]), "verified_pngs": len(rows),
                      "archive_sha256": archive_sha, "aggregate_manifest_written": True}))


if __name__ == "__main__":
    main()
