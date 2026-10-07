"""Recheck public aggregate arithmetic without data access, inference or training.

Run with Python 3.10+; only the standard library is required. The input files
contain aggregate measurements and counts, never source image/frame identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(folder: Path, name: str) -> dict:
    value = json.loads((folder / name).read_text(encoding="utf-8"))
    require(value.get("schema_version") == 1, f"Unsupported schema: {name}")
    return value


def codec_distance(video: dict, photo: dict, weights: dict) -> float:
    """The exploratory weighted distance recorded in codec_criterion.json."""
    return (
        weights["flat_edge_log_ratio_weight"]
        * abs(math.log(video["flat_edge"] / photo["flat_edge"]))
        + weights["zero_absolute_difference_weight"]
        * abs(video["zero"] - photo["zero"])
        + weights["block8_log_ratio_weight"]
        * abs(math.log(video["block8"] / photo["block8"]))
        + weights["hcorr_absolute_difference_weight"]
        * abs(video["hcorr"] - photo["hcorr"])
    )


def check_codec(folder: Path) -> dict:
    criterion = load_json(folder, "codec_criterion.json")
    require(criterion["distance"]["logarithm"] == "natural", "Expected natural logarithms")
    metrics = ("flat_e3", "zero", "edge_e3", "flat_edge", "block8", "hcorr")
    records = {}
    with (folder / criterion["input"]).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            group = row["group"]
            require(group not in records, f"Duplicate codec group: {group}")
            require(re.fullmatch(r"(?:video_[ABC]|photo_(?:png|jpg\d+|crf\d+))", group) is not None,
                    "Unexpected codec group alias")
            values = {key: float(row[key]) for key in metrics}
            require(all(math.isfinite(v) for v in values.values()), f"Non-finite codec value: {group}")
            require(values["flat_edge"] > 0 and values["block8"] > 0,
                    f"Logarithmic codec metrics must be positive: {group}")
            require(0 <= values["zero"] <= 1 and -1 <= values["hcorr"] <= 1,
                    f"Codec fraction/correlation outside bounds: {group}")
            records[group] = {"n": int(row["n"]), **values}
    expected_photos = {"photo_png"}
    expected_photos.update(f"photo_jpg{q}" for q in criterion["encoding_settings"]["jpeg_qualities"])
    expected_photos.update(f"photo_crf{q}" for q in criterion["encoding_settings"]["h264_crfs"])
    require(set(records) == expected_photos | {"video_A", "video_B", "video_C"},
            "Codec table does not contain exactly the recorded groups/settings")
    require(all(row["n"] == (100 if group.startswith("video_") else 40)
                for group, row in records.items()), "Unexpected codec sample count")
    rankings = {}
    for video in ("video_A", "video_B", "video_C"):
        ranking = sorted(
            ({"variant": p, "distance": codec_distance(records[video], records[p], criterion["distance"])}
             for p in sorted(expected_photos)),
            key=lambda row: (row["distance"], row["variant"]),
        )
        expected = criterion["reference_result"]
        require(ranking[0]["variant"] == expected["nearest_variant_each_video"][video],
                f"Codec nearest variant changed: {video}")
        require(math.isclose(ranking[0]["distance"], expected["nearest_distances"][video],
                             rel_tol=1e-12, abs_tol=1e-12),
                f"Codec reference distance changed: {video}")
        rankings[video] = ranking
    return {"rankings": rankings, "interpretation": "Exploratory criterion on recorded aggregate medians"}


def check_photo_split(folder: Path) -> dict:
    data = load_json(folder, "photo_split.json")
    splits = ("train", "validation", "heldout_test")
    summed = {split: 0 for split in splits}
    total = 0
    camera_total = 0
    for source in data["sources"]:
        for unit in ("originals", "groups"):
            counts = source[unit]
            require(all(isinstance(counts[s], int) and counts[s] >= 0 for s in (*splits, "total")),
                    f"Invalid {unit} count: {source['source']}")
            require(sum(counts[s] for s in splits) == counts["total"],
                    f"Partition does not sum to total: {source['source']}/{unit}")
        originals = source["originals"]
        total += originals["total"]
        if source["source"] != "reviewed_4K_frames":
            camera_total += originals["total"]
        for split in splits:
            summed[split] += originals[split]
    totals = data["totals"]
    require(total == totals["anchor_originals"] and camera_total == totals["camera_photos"],
            "Photo/anchor original totals disagree")
    require(summed == {"train": totals["train_originals"],
                       "validation": totals["validation_pool_originals"],
                       "heldout_test": totals["heldout_test_pool_originals"]},
            "Source counts disagree with pooled split counts")
    cap = data["evaluation_cap_originals_per_split"]
    indexed = {"train": summed["train"],
               "validation": min(cap, summed["validation"]),
               "heldout_test": min(cap, summed["heldout_test"])}
    require(indexed == data["indexed_originals"], "Indexed counts disagree with cap arithmetic")
    require(data["recipe_rows_per_indexed_original"] == {"sharp": 1, "blurred": 1},
            "Expected one recipe row per class per original")
    require(indexed == data["rows_per_class"], "Paired recipe row counts disagree")
    omitted = summed["validation"] + summed["heldout_test"] - indexed["validation"] - indexed["heldout_test"]
    require(omitted == data["heldout_originals_omitted_after_caps"], "Omitted-original count disagrees")
    require(data["omitted_originals_returned_to_training"] == 0, "Omitted held-out originals returned to train")
    frame = data["frame_snapshot"]
    require(frame["processed_video_files"] - frame["duplicate_copy_files_excluded_by_selector"]
            == frame["distinct_video_groups"], "Video file/group arithmetic disagrees")
    require(sum(frame["video_groups"].values()) == frame["distinct_video_groups"],
            "Video split groups do not sum to the group total")
    require(frame["eligible_labelled_frames_in_selection_summary"]
            <= frame["reported_teacher_labelled_frames_before_selection"],
            "Eligible frame count exceeds the labelled-frame count")
    for split in ("validation", "heldout_test"):
        require(frame[f"selected_{split}"]["sharp"] == frame[f"selected_{split}"]["blurred"],
                f"Frame class counts are unbalanced: {split}")
    return {"anchor_originals": total, "camera_photos": camera_total,
            "original_split_pools": summed, "indexed_originals": indexed,
            "omitted_heldout_originals": omitted,
            "interpretation": "Aggregate partition and cap arithmetic; not recovery of private group assignments"}


def check_provenance(folder: Path) -> dict:
    data = load_json(folder, "run_provenance.json")
    require(len(data["runs"]) == 3, "Expected three published-run summaries")
    expected_inputs = {"RGB", "signed_mesh_residual", "grayscale"}
    require({run["input"] for run in data["runs"]} == expected_inputs, "Unexpected run input summaries")
    for run in data["runs"]:
        for key in ("index_sha256", "frame_selection_sha256", "code_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", run[key]) is not None, f"Invalid digest: {key}")
        require(run["index_sha256"] == data["shared_input_index_sha256"], "Run index identifiers disagree")
        require(run["frame_selection_sha256"] == data["shared_frame_selection_sha256"],
                "Run frame-selection identifiers disagree")
        require(run["epochs"] == 100 and run["resume_record_count"] == 2, "Unexpected run summary counts")
        require(run["all_resume_code_digests_match_start"] is True
                and run["local_audit_index_and_selection_match_start"] is True,
                "A run summary records an integrity mismatch")
    require(len({run["seed"] for run in data["runs"]}) == 1, "Recorded run seeds disagree")
    return {"run_count": len(data["runs"]),
            "shared_input_index_sha256": data["shared_input_index_sha256"],
            "shared_frame_selection_sha256": data["shared_frame_selection_sha256"],
            "interpretation": "Consistency of released records; private source files are not independently rehashed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", help="Print all recomputed rankings and aggregate checks")
    args = parser.parse_args()
    try:
        codec = check_codec(args.data_dir)
        photos = check_photo_split(args.data_dir)
        provenance = check_provenance(args.data_dir)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"Evidence check failed: {error}\n")
    if args.json:
        print(json.dumps({"codec": codec, "photo_split": photos, "recorded_provenance": provenance}, indent=2))
        return
    print("Codec criterion: CRF 23 ranks first for all three recorded video groups.")
    for video, ranking in codec["rankings"].items():
        print(f"  {video}: D = {ranking[0]['distance']:.9f}")
    print(f"Photo split: {photos['camera_photos']:,} camera photos plus 231 reviewed 4K frames "
          f"= {photos['anchor_originals']:,} originals.")
    print(f"  {photos['indexed_originals']['train']:,} training originals; 2,000 indexed originals "
          f"per held-out split; {photos['omitted_heldout_originals']:,} other held-out originals omitted.")
    print("Recorded provenance: three run summaries agree on the input-index and frame-selection digests.")
    print("Checks cover aggregate arithmetic and record consistency; raw measurements, pixels and training are not reproduced.")


if __name__ == "__main__":
    main()
