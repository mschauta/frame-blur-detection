"""Paired video-cluster bootstrap of saved validation scores; no model or image reads.

Requires NumPy. Inputs stay private; the output contains aggregates and hashes only.
Only frame rows in the index's validation split are used. Never opens test scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RECALLS = {"r90": 0.90, "r95": 0.95, "r98": 0.98}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["source"] == "frames" and row["split"] == "val"]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Validation frame IDs must be nonempty and unique")
    labels = np.asarray([int(row["target"]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Both binary classes are required")
    private_groups = sorted({row["group"] for row in rows})
    group_number = {group: number for number, group in enumerate(private_groups)}
    groups = np.asarray([group_number[row["group"]] for row in rows])
    return ids, labels, groups, len(private_groups)


def load_scores(path: Path, ids: list[str], labels: np.ndarray):
    scores = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") != "frames":
                continue
            key = row["sample_id"]
            if key in scores:
                raise ValueError("Duplicate frame prediction")
            scores[key] = (int(row["target"]), float(row["p_blurred"]))
    if set(scores) != set(ids):
        raise ValueError("Prediction frame IDs differ from the validation cohort")
    if any(scores[key][0] != int(label) for key, label in zip(ids, labels)):
        raise ValueError("Prediction labels differ from the validation index")
    result = np.asarray([scores[key][1] for key in ids])
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Scores must be finite and within [0, 1]")
    return result


def pair_counts(y: np.ndarray, p: np.ndarray, g: np.ndarray, count: int):
    """Concordant positive/negative pairs per group, with half credit for ties."""
    result = np.zeros((count, count))
    for positive_group in range(count):
        positive = p[(y == 1) & (g == positive_group)]
        for negative_group in range(count):
            negative = np.sort(p[(y == 0) & (g == negative_group)])
            left = np.searchsorted(negative, positive, side="left")
            right = np.searchsorted(negative, positive, side="right")
            result[positive_group, negative_group] = ((left + right) / 2).sum()
    return result


def replicated_quantile(sorted_scores, sorted_groups, multiplicities, q):
    """NumPy's linear quantile of a conceptual array with whole groups repeated."""
    cumulative = np.cumsum(multiplicities[sorted_groups])
    size = int(cumulative[-1])
    if size == 0:
        return float("nan")
    position = (size - 1) * q
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    a = sorted_scores[np.searchsorted(cumulative, lower, side="right")]
    b = sorted_scores[np.searchsorted(cumulative, upper, side="right")]
    return float(a + (b - a) * (position - lower))


def verify_math():
    """Check ties, repeated/omitted clusters and quantile interpolation independently."""
    y = np.asarray([0, 1, 0, 1, 0, 1])
    p = np.asarray([0.2, 0.2, 0.4, 0.9, 0.8, 0.8])
    g = np.asarray([0, 0, 1, 1, 2, 2])
    pairs = pair_counts(y, p, g, 3)
    for w in (np.ones(3, dtype=int), np.asarray([2, 0, 1])):
        i = np.repeat(np.arange(len(y)), w[g])
        pos, neg = p[i][y[i] == 1], p[i][y[i] == 0]
        direct = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))
        grouped = (w @ pairs @ w) / (len(pos) * len(neg))
        if not np.isclose(direct, grouped, atol=1e-12):
            raise AssertionError("Grouped AUC verification failed")
        order = np.argsort(p)
        for q in (0.0, 0.02, 0.05, 0.1, 0.5, 1.0):
            actual = replicated_quantile(p[order], g[order], w, q)
            if not np.isclose(actual, np.quantile(p[i], q), atol=1e-12):
                raise AssertionError("Replicated quantile verification failed")


def estimate(y, p, g, count, draws):
    n0 = np.bincount(g[y == 0], minlength=count)
    n1 = np.bincount(g[y == 1], minlength=count)
    total0, total1 = draws @ n0, draws @ n1
    if (total0 == 0).any() or (total1 == 0).any():
        raise ValueError("A bootstrap draw omitted an entire class")
    pair = pair_counts(y, p, g, count)
    auc = np.einsum("bi,ij,bj->b", draws, pair, draws) / (total0 * total1)
    positive_order = np.argsort(p[y == 1], kind="stable")
    positive = p[y == 1][positive_order]
    positive_groups = g[y == 1][positive_order]
    fixed, refit, thresholds = {}, {}, {}
    for name, recall in RECALLS.items():
        threshold = float(np.quantile(positive, 1 - recall))
        thresholds[name] = threshold
        kept = np.bincount(g[(y == 0) & (p < threshold)], minlength=count)
        caught = np.bincount(g[(y == 1) & (p >= threshold)], minlength=count)
        fixed[f"sharp_kept_{name}"] = draws @ kept / total0
        fixed[f"blurred_caught_{name}"] = draws @ caught / total1
        refit_kept, refit_caught = np.zeros(len(draws)), np.zeros(len(draws))
        for index, weights in enumerate(draws):
            threshold_ = replicated_quantile(positive, positive_groups, weights, 1 - recall)
            frame_weights = weights[g]
            refit_kept[index] = frame_weights[(y == 0) & (p < threshold_)].sum() / total0[index]
            refit_caught[index] = frame_weights[(y == 1) & (p >= threshold_)].sum() / total1[index]
        refit[f"sharp_kept_{name}"] = refit_kept
        refit[f"blurred_caught_{name}"] = refit_caught
    return {"auc": auc}, fixed, refit, thresholds


def intervals(values):
    return {key: {"estimate": float(value[0]),
                  "ci95": [float(x) for x in np.quantile(value[1:], [0.025, 0.975])]}
            for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True,
                        help="MODEL=CSV; repeat for models evaluated on identical validation frames")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error("Use at least 1000 draws")
    verify_math()
    ids, labels, groups, count = load_index(args.index)
    rng = np.random.default_rng(args.seed)
    # Row zero is the original cohort; remaining rows resample count clusters.
    picks = rng.integers(count, size=(args.draws, count))
    draws = np.vstack([np.ones(count, dtype=int),
                       np.asarray([np.bincount(row, minlength=count) for row in picks])])
    output = {
        "analysis": "Paired validation video-cluster percentile bootstrap",
        "checkpoint_epoch": 99,
        "cluster_count": count,
        "frames": len(ids),
        "sharp_frames": int((labels == 0).sum()),
        "blurred_frames": int((labels == 1).sum()),
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "confidence": 0.95,
        "software": {"numpy": np.__version__},
        "input_sha256": {"index": digest(args.index)},
        "script_sha256": digest(Path(__file__)),
        "labels": "filtered frozen-teacher labels; agreement, not objective accuracy",
        "sampling": f"{count} groups sampled uniformly with replacement; all frames retained; pooled frame-weighted metrics",
        "limitations": [
            "Only video sampling uncertainty conditional on this labelled cohort, split and one trained checkpoint per input.",
            "Does not measure training-seed variation, label error, source selection bias or performance outside this domain.",
            "Only 11 video groups; percentile intervals may have limited coverage.",
            "Validation was repeatedly used during model development; these are not untouched test intervals.",
            "Saved sigmoid scores are rounded to six decimal places; ties and point estimates can differ from full precision.",
            "Fixed-threshold intervals condition on the original validation-derived thresholds.",
            "Recalibrated intervals refit each threshold within each bootstrap draw; target recall is imposed on that draw, not guaranteed on new videos."
        ],
        "models": {},
        "paired_differences": {}
    }
    arrays = {}
    for item in args.predictions:
        name, separator, raw_path = item.partition("=")
        if not separator or name not in {"rgb", "gray", "p99"} or name in arrays:
            parser.error("Use each model name rgb, gray or p99 at most once as MODEL=CSV")
        path = Path(raw_path)
        p = load_scores(path, ids, labels)
        auc, fixed, refit, thresholds = estimate(labels, p, groups, count, draws)
        output["input_sha256"][name] = digest(path)
        output["models"][name] = {"auc": intervals(auc)["auc"],
                                 "thresholds_from_saved_validation_scores": thresholds,
                                 "fixed_threshold": intervals(fixed),
                                 "recalibrated_threshold": intervals(refit)}
        arrays[name] = {"auc": auc, "fixed_threshold": fixed, "recalibrated_threshold": refit}
    ordered = [name for name in ("rgb", "gray", "p99") if name in arrays]
    for i, first in enumerate(ordered):
        for second in ordered[i + 1:]:
            diff = {}
            for mode in arrays[first]:
                difference = {key: value - arrays[second][mode][key]
                              for key, value in arrays[first][mode].items()}
                diff[mode] = intervals(difference)
            output["paired_differences"][f"{first}_minus_{second}"] = diff
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation_epoch099_bootstrap.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"models": list(output["models"]), "frames": len(ids),
                      "groups": count, "draws": args.draws, "aggregate_output_written": True}))


if __name__ == "__main__":
    main()
