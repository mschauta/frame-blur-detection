"""Analyse fixed-checkpoint held-out scores; inputs remain private.

No images, model weights or GPU are needed. Thresholds must come from validation,
never from the test scores. Requires Python 3.10+ and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

OPERATING_POINTS = ("r90", "r95", "r98")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_cohort(path):
    rows = [r for r in read_csv(path) if r["split"] == "test"]
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test sample IDs")
    if len(rows) != 10592 or {r["source"] for r in rows} != {"frames", "photo"}:
        raise ValueError("This report requires the frozen 10,592-sample test cohort")
    for source, size in (("frames", 6592), ("photo", 4000)):
        labels = [int(r["target"]) for r in rows if r["source"] == source]
        if len(labels) != size or labels.count(0) != size // 2 or labels.count(1) != size // 2:
            raise ValueError("Frozen test cohort class counts do not match")
    frames = [r for r in rows if r["source"] == "frames"]
    groups = sorted({r["group"] for r in frames})
    if len(groups) != 7:
        raise ValueError("Exactly seven test video groups are required")
    for group in groups:
        labels = [int(r["target"]) for r in frames if r["group"] == group]
        if set(labels) != {0, 1}:
            raise ValueError("Every video group must contain both classes")
    photos = [r for r in rows if r["source"] == "photo"]
    originals = sorted({r["png"] for r in photos})
    if len(originals) != 2000:
        raise ValueError("Exactly 2,000 paired photo originals are required")
    photo_labels = {}
    for r in photos:
        photo_labels.setdefault(r["png"], []).append(int(r["target"]))
    if any(sorted(labels) != [0, 1] for labels in photo_labels.values()):
        raise ValueError("Each photo original must have one sharp and one blurred recipe")
    return rows, groups, originals


def load_scores(path, rows):
    parsed = read_csv(path)
    ids = [r["sample_id"] for r in parsed]
    if len(ids) != len(set(ids)) or set(ids) != {r["sample_id"] for r in rows}:
        raise ValueError("Score IDs must match the frozen test cohort exactly")
    lookup = {r["sample_id"]: r for r in parsed}
    values = []
    for expected in rows:
        actual = lookup[expected["sample_id"]]
        if actual["source"] != expected["source"] or int(actual["target"]) != int(expected["target"]):
            raise ValueError("Score source/label differs from the frozen index")
        values.append(float(actual["p_blurred"]))
    scores = np.asarray(values)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite probabilities within [0, 1]")
    return scores


def verify_score_provenance(path, manifest_sha, checkpoint, expected_samples):
    """Accept only complete score files bound to the unchanged locked manifest."""
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError("Every score CSV requires its inference sidecar")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
    if sidecar.get("complete") is not True:
        raise ValueError("Inference sidecar does not declare complete predictions")
    if sidecar.get("manifest_sha256") != manifest_sha:
        raise ValueError("Score sidecar belongs to a different locked inference manifest")
    score_sha = digest(path)
    if sidecar.get("scores_sha256") != score_sha:
        raise ValueError("Score CSV hash differs from its inference sidecar")
    if sidecar.get("samples") != expected_samples:
        raise ValueError("Inference sidecar sample count differs from the frozen cohort")
    if sidecar.get("weights_sha256") != checkpoint["weights_sha256"]:
        raise ValueError("Inference sidecar weight hash differs from the fixed checkpoint")
    return score_sha, digest(sidecar_path)


def pair_counts(y, p, g, count):
    """Positive/negative concordant pair counts, with half credit for score ties."""
    pair = np.zeros((count, count))
    for a in range(count):
        positive = p[(y == 1) & (g == a)]
        for b in range(count):
            negative = np.sort(p[(y == 0) & (g == b)])
            left = np.searchsorted(negative, positive, side="left")
            right = np.searchsorted(negative, positive, side="right")
            pair[a, b] = ((left + right) / 2).sum()
    return pair


def weighted_auc(y, bins, groups, multiplicities):
    """AUC of the conceptual sample obtained by repeating entire clusters.

    `bins` is the ascending tied-score bin of each sample. This alternative to
    a cluster-pair matrix also supports the 2,000 paired-photo clusters.
    """
    weights = multiplicities[groups]
    n0, n1 = weights[y == 0].sum(), weights[y == 1].sum()
    if n0 == 0 or n1 == 0:
        return np.nan
    count = int(bins.max()) + 1
    negative = np.bincount(bins[y == 0], weights=weights[y == 0], minlength=count)
    positive = np.bincount(bins[y == 1], weights=weights[y == 1], minlength=count)
    return float((positive * (np.cumsum(negative) - negative / 2)).sum() / (n0 * n1))


def divide(numerator, denominator):
    out = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    return np.divide(numerator, denominator, out=out, where=np.asarray(denominator) != 0)


def estimates(y, p, g, count, draws, thresholds, photo=False, equal_group=True):
    n0 = np.bincount(g[y == 0], minlength=count)
    n1 = np.bincount(g[y == 1], minlength=count)
    total0, total1 = draws @ n0, draws @ n1
    if photo:
        _, bins = np.unique(p, return_inverse=True)
        auc = np.asarray([weighted_auc(y, bins, g, w) for w in draws])
    else:
        pair = pair_counts(y, p, g, count)
        auc = divide(np.einsum("bi,ij,bj->b", draws, pair, draws), total0 * total1)
    result = {"auc": auc}
    for name in OPERATING_POINTS:
        threshold = thresholds[name]
        kept = np.bincount(g[(y == 0) & (p < threshold)], minlength=count)
        caught = np.bincount(g[(y == 1) & (p >= threshold)], minlength=count)
        result[f"sharp_kept_{name}"] = divide(draws @ kept, total0)
        result[f"blurred_caught_{name}"] = divide(draws @ caught, total1)
        if equal_group:
            # Each sampled group has equal weight, instead of each frame.
            if (n0 == 0).any() or (n1 == 0).any():
                raise ValueError("Equal-group operating points require both classes in each group")
            result[f"equal_group_sharp_kept_{name}"] = divide(draws @ (kept / n0), draws.sum(axis=1))
            result[f"equal_group_blurred_caught_{name}"] = divide(draws @ (caught / n1), draws.sum(axis=1))
    return result


def intervals(values):
    output = {}
    for key, values_ in values.items():
        values_ = np.asarray(values_)
        valid = values_[1:][np.isfinite(values_[1:])]
        output[key] = {
            "estimate": float(values_[0]) if np.isfinite(values_[0]) else None,
            "ci95": [float(x) for x in np.quantile(valid, [0.025, 0.975])] if len(valid) else None,
            "valid_bootstrap_draws": int(len(valid)),
        }
    return output


def point_stats(y, p, g, count, weights, thresholds):
    values = estimates(y, p, g, count, weights[np.newaxis], thresholds, equal_group=False)
    return {k: float(v[0]) if np.isfinite(v[0]) else None for k, v in values.items()}


def verify_math():
    """Independent expanded-array checks cover tied scores and repeated clusters."""
    y = np.asarray([0, 1, 0, 1, 0, 1])
    p = np.asarray([0.2, 0.2, 0.4, 0.9, 0.8, 0.8])
    g = np.asarray([0, 0, 1, 1, 2, 2])
    _, bins = np.unique(p, return_inverse=True)
    pairs = pair_counts(y, p, g, 3)
    thresholds = {name: 0.4 for name in OPERATING_POINTS}
    for weights in (np.ones(3, dtype=int), np.asarray([2, 0, 1]), np.asarray([0, 3, 0])):
        i = np.repeat(np.arange(len(y)), weights[g])
        pos, neg = p[i][y[i] == 1], p[i][y[i] == 0]
        direct = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))
        grouped = (weights @ pairs @ weights) / (len(pos) * len(neg))
        actual = weighted_auc(y, bins, g, weights)
        if not np.allclose([grouped, actual], direct, rtol=0, atol=1e-12):
            raise AssertionError("Grouped AUC differs from the expanded-array example")
        stats = estimates(y, p, g, 3, weights[np.newaxis], thresholds)
        if not np.isclose(stats["sharp_kept_r95"][0], np.mean(neg < 0.4)):
            raise AssertionError("Weighted sharp retention check failed")
        if not np.isclose(stats["blurred_caught_r95"][0], np.mean(pos >= 0.4)):
            raise AssertionError("Weighted blurred recall check failed")


def bootstrap_draws(count, draws, rng):
    picks = rng.integers(count, size=(draws, count))
    return np.vstack([np.ones(count, dtype=int),
                      np.asarray([np.bincount(pick, minlength=count) for pick in picks])])


def confidence_mask(path, rows):
    parsed = [r for r in read_csv(path) if r["split"] == "test"]
    ids = [r["sample_id"] for r in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate measurement-agreement subset IDs")
    cohort = {r["sample_id"]: r for r in rows if r["source"] == "frames"}
    if not set(ids).issubset(cohort):
        raise ValueError("Measurement-agreement subset is outside the frozen test frames")
    for r in parsed:
        expected_label = "blurred" if int(cohort[r["sample_id"]]["target"]) else "sharp"
        if r["confident"] != expected_label:
            raise ValueError("Measurement-agreement label differs from the teacher index")
    retained = set(ids)
    return np.asarray([r["sample_id"] in retained for r in rows if r["source"] == "frames"])


def format_interval(metric, percent=False, precision=None, percentage_points=False):
    scale = 100 if percent or percentage_points else 1
    if precision is None:
        precision = 1 if percent or percentage_points else 3
    point, ci = metric["estimate"], metric["ci95"]
    if point is None:
        return "undefined"
    text = f"{point * scale:.{precision}f}"
    if ci is not None:
        text += f" [{ci[0] * scale:.{precision}f}, {ci[1] * scale:.{precision}f}]"
    return text + (" pp" if percentage_points else "%" if percent else "")


def markdown(output):
    text = ["# Seven-group held-out evaluation", "",
            "All checkpoints and operating thresholds were fixed before this calculation. The primary comparisons use epoch 99; earlier externally selected checkpoints are secondary. Test scores are never used to adjust thresholds or choose a checkpoint.", "",
            "Frame metrics measure agreement with filtered frozen-teacher labels, not objective blur accuracy. The cohort has 6,592 frames (3,296 per class) from seven video groups and 4,000 synthetic photo recipes (2,000 paired originals). Photo sharp labels are assigned by construction and do not establish objective sharpness.", "",
            f"The largest video group supplies {output['cohort']['largest_video_frame_share'] * 100:.1f}% of test frames. Pooled metrics therefore mainly describe the composition of this cohort. Equal-group operating points and seven leave-one-group-out analyses accompany the pooled estimates.", "",
            "## Primary results", "",
            "Values are point estimates with 95% percentile bootstrap intervals. r95 denotes a validation recall target; the realised test blurred recall is measured separately.", "",
            "| Input | Frame AUC | Sharp kept at fixed r95 threshold | Blurred caught at fixed r95 threshold |",
            "|---|---|---|---|"]
    primary = [key for key, item in output["models"].items() if item["selection_role"] == "primary"]
    for key in primary:
        model = output["models"][key]
        stats = model["video_cluster_bootstrap"]
        text.append(f"| {model['input']} (epoch {model['epoch']}) | {format_interval(stats['auc'])} | {format_interval(stats['sharp_kept_r95'], True)} | {format_interval(stats['blurred_caught_r95'], True)} |")
    text += ["", "## Equal-group operating points", "",
             "Each video group receives equal weight in this table. These are means of within-group class-specific rates, not an alternative pooled AUC.", "",
             "| Input | Mean sharp kept, r95 | Mean blurred caught, r95 |", "|---|---|---|"]
    for key in primary:
        model = output["models"][key]
        stats = model["video_cluster_bootstrap"]
        text.append(f"| {model['input']} | {format_interval(stats['equal_group_sharp_kept_r95'], True)} | {format_interval(stats['equal_group_blurred_caught_r95'], True)} |")
    text += ["", "## Paired primary differences", "",
             "The same video-group draw is used for every checkpoint. Differences are first model minus second model; retention and recall differences are percentage points. An interval containing zero does not establish equivalence.", "",
             "| Comparison | AUC difference | Sharp-kept difference, r95 | Blurred-caught difference, r95 |", "|---|---|---|---|"]
    for key, stats in output["paired_primary_differences"].items():
        text.append(f"| {key.replace('_minus_', ' minus ')} | {format_interval(stats['auc'])} | {format_interval(stats['sharp_kept_r95'], percentage_points=True)} | {format_interval(stats['blurred_caught_r95'], percentage_points=True)} |")
    text += ["", "## Secondary checkpoints and photo recipes", "",
             "These rows do not change the primary epoch-99 comparison. Photo uncertainty resamples paired originals; it does not account for dependence between originals in the same source album.", "",
             "Photo AUCs are rounded to six decimal places; a displayed 1.000000 can be a rounded value. The JSON retains full precision.", "",
             "| Checkpoint | Role | Frame AUC | Photo AUC |", "|---|---|---|---|"]
    for key, model in output["models"].items():
        text.append(f"| {key} | {model['selection_role']} | {format_interval(model['video_cluster_bootstrap']['auc'])} | {format_interval(model['photo_original_bootstrap']['auc'], precision=6)} |")
    if output["cohort"].get("measurement_agreement"):
        n = output["cohort"]["measurement_agreement"]
        text += ["", "## Measurement-agreement subset", "",
                 f"This subset contains {n['samples']} frames ({n['sharp']} sharp and {n['blurred']} blurred). Labels agree with the independent classical measurement, which filters the teacher-labelled cohort; this does not create independent human ground truth.", "",
                 "| Checkpoint | Subset AUC |", "|---|---|"]
        for key, model in output["models"].items():
            text.append(f"| {key} | {format_interval(model['measurement_agreement_bootstrap']['auc'])} |")
    text += ["", "## Interpretation and limits", ""]
    text += [f"- {item}" for item in output["limitations"]]
    text += ["", "The full thresholds, anonymous group counts, per-group rates, leave-one-group-out results, paired differences and file hashes are in [heldout_results.json](heldout_results.json). Private frame IDs, video names, source paths and raw scores are not distributed.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--confidence", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    verify_math()
    if args.self_test:
        print("Grouped AUC, tied-score weighted AUC and fixed-threshold expanded-array checks passed")
        return
    if not all((args.index, args.scores, args.manifest, args.out)):
        parser.error("--index, --scores, --manifest and --out are required")
    if args.draws < 1000:
        parser.error("Use at least 1,000 bootstrap draws")
    manifest_sha = digest(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("prediction_parameters_locked") is not True:
        raise ValueError("Prediction parameters must be locked before analysing test scores")
    if manifest["index_sha256"] != digest(args.index):
        raise ValueError("Frozen index hash differs from the inference manifest")
    rows, groups, originals = load_cohort(args.index)
    source = np.asarray([r["source"] for r in rows])
    y = np.asarray([int(r["target"]) for r in rows])
    frame_mask, photo_mask = source == "frames", source == "photo"
    group_lookup = {key: i for i, key in enumerate(groups)}
    original_lookup = {key: i for i, key in enumerate(originals)}
    g = np.asarray([group_lookup[r["group"]] for r in rows if r["source"] == "frames"])
    pg = np.asarray([original_lookup[r["png"]] for r in rows if r["source"] == "photo"])
    yf, yp = y[frame_mask], y[photo_mask]
    rng = np.random.default_rng(args.seed)
    draws = bootstrap_draws(7, args.draws, rng)
    photo_draws = bootstrap_draws(2000, args.draws, rng)
    agreement = confidence_mask(args.confidence, rows) if args.confidence else None
    group_counts = [{"group": f"video_{i + 1:02d}", "frames": int((g == i).sum()),
                     "sharp": int(((g == i) & (yf == 0)).sum()),
                     "blurred": int(((g == i) & (yf == 1)).sum())} for i in range(7)]
    output = {
        "analysis": "Fixed-checkpoint held-out test; paired video-cluster percentile bootstrap",
        "bootstrap_draws": args.draws, "seed": args.seed, "confidence": 0.95,
        "software": {"numpy": np.__version__},
        "input_sha256": {"index": digest(args.index), "manifest": manifest_sha},
        "script_sha256": digest(Path(__file__)),
        "cohort": {"frames": 6592, "sharp_frames": 3296, "blurred_frames": 3296,
                   "video_groups": 7, "photo_recipes": 4000, "photo_originals": 2000,
                   "video_group_counts": group_counts,
                   "largest_video_frame_share": max(item["frames"] for item in group_counts) / 6592},
        "labels": {"frames": "Filtered frozen-teacher agreement, not objective blur accuracy",
                   "photos": "Sharp/blurred recipe labels assigned by construction"},
        "threshold_rule": "Use saved validation thresholds without fitting to test or bootstrap samples; p >= threshold is blurred",
        "sampling": "Seven whole video groups sampled uniformly with replacement; 2,000 whole photo-original pairs sampled uniformly with replacement; identical draws across models",
        "limitations": [
            "Only one training run per input and seven test video groups are represented; no training-seed uncertainty is estimated.",
            "Percentile bootstrap coverage is uncertain with seven imbalanced groups; intervals describe conditional group sampling, not all possible deployment domains.",
            "A validation recall target is not guaranteed on held-out groups; report realised test recall at the unchanged threshold.",
            "Teacher agreement and agreement with a classical measurement do not establish objective human-labelled blur accuracy.",
            "Synthetic photo recipe labels, blur lengths and original-pair uncertainty do not validate separation of native blur types; source-album dependence is not included in the photo intervals.",
            "The earlier checkpoints were selected using external evaluation/selection sets; this report does not remove that selection dependence.",
            "Paired difference intervals are descriptive; they are not equivalence tests, and no multiple-comparison adjustment is applied.",
            "The frozen cohort and deterministic synthetic recipes are evaluated with the recorded software and weights; no claim of immutable historical source pixels is made."
        ],
        "models": {}, "paired_primary_differences": {},
    }
    if agreement is not None:
        output["input_sha256"]["measurement_agreement"] = digest(args.confidence)
        output["cohort"]["measurement_agreement"] = {"samples": int(agreement.sum()),
                                                        "sharp": int((yf[agreement] == 0).sum()),
                                                        "blurred": int((yf[agreement] == 1).sum())}
    arrays = {}
    allowed_ids = {"rgb_ep099", "gray_ep099", "p99_ep099", "rgb_ep073", "gray_ep058", "p99_ep053"}
    checkpoints = manifest["checkpoints"]
    if {ck["id"] for ck in checkpoints} != allowed_ids or len(checkpoints) != 6:
        raise ValueError("Exactly the six fixed study checkpoints are required")
    for checkpoint in checkpoints:
        key = checkpoint["id"]
        if checkpoint["input"] not in {"rgb", "gray", "p99"}:
            raise ValueError("Input must be a public study representation name")
        expected_key = f"{checkpoint['input']}_ep{int(checkpoint['epoch']):03d}"
        if expected_key != key:
            raise ValueError("Checkpoint ID differs from input and epoch")
        weight_sha = checkpoint["weights_sha256"]
        if len(weight_sha) != 64 or any(c not in "0123456789abcdef" for c in weight_sha):
            raise ValueError("A SHA-256 weight hash is required")
        if checkpoint["selection_role"] not in {"primary", "secondary"}:
            raise ValueError("Selection role must be primary or secondary")
        if (checkpoint["epoch"] == 99) != (checkpoint["selection_role"] == "primary"):
            raise ValueError("Only fixed epoch-99 checkpoints are primary")
        thresholds = {name: float(checkpoint["thresholds"][name]) for name in OPERATING_POINTS}
        if not all(np.isfinite(t) and 0 <= t <= 1 for t in thresholds.values()):
            raise ValueError("Thresholds must be finite probabilities")
        path = args.scores / f"{key}.csv"
        sha, sidecar_sha = verify_score_provenance(path, manifest_sha, checkpoint, len(rows))
        p = load_scores(path, rows)
        pf, pp = p[frame_mask], p[photo_mask]
        values = estimates(yf, pf, g, 7, draws, thresholds)
        arrays[key] = values
        result = {name: checkpoint[name] for name in ("input", "epoch", "selection_role", "weights_sha256")}
        result["validation_thresholds"] = thresholds
        result["score_sha256"] = sha
        result["score_sidecar_sha256"] = sidecar_sha
        result["video_cluster_bootstrap"] = intervals(values)
        result["photo_original_bootstrap"] = intervals(estimates(yp, pp, pg, 2000, photo_draws, thresholds, photo=True, equal_group=False))
        result["per_video"] = []
        result["leave_one_video_out"] = []
        for i in range(7):
            w = np.zeros(7, dtype=int); w[i] = 1
            result["per_video"].append({"group": f"video_{i + 1:02d}", **point_stats(yf, pf, g, 7, w, thresholds)})
            w = np.ones(7, dtype=int); w[i] = 0
            result["leave_one_video_out"].append({"excluded_group": f"video_{i + 1:02d}",
                                                  "remaining_frames": int((g != i).sum()),
                                                  **point_stats(yf, pf, g, 7, w, thresholds)})
        if agreement is not None:
            result["measurement_agreement_bootstrap"] = intervals(estimates(yf[agreement], pf[agreement], g[agreement], 7, draws, thresholds, equal_group=False))
        output["models"][key] = result
    primary = [ck["id"] for ck in checkpoints if ck["selection_role"] == "primary"]
    for first, second in itertools.combinations(primary, 2):
        output["paired_primary_differences"][f"{first}_minus_{second}"] = intervals({
            key: arrays[first][key] - arrays[second][key] for key in arrays[first]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heldout_results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "RESULTS.md").write_text(markdown(output), encoding="utf-8")
    print(json.dumps({"aggregate_output_written": True, "checkpoints": len(checkpoints), "frames": 6592,
                      "video_groups": 7, "photo_recipes": 4000, "draws": args.draws}))


if __name__ == "__main__":
    main()
