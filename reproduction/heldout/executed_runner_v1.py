"""Prepare a fixed private test cache, then score six public checkpoints.

Requires the original trainer's data/synth modules and its Python environment.
Preparation uses CPU only; inference requires a separately available GPU window.
All inputs, caches and per-image predictions stay in the chosen private output.
The public aggregate report is produced by analyse_test.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_INDEX = "1d82624aad595396f1943c80cc2acf520afd655ae2cd4f168c20047d2b548810"
EXPECTED_SELECTION = "5b2f70e6515b7b8c682d746b17e28ea9304fab20d011ae0fd63b6d394bcc7003"
CHECKPOINTS = [("rgb", 99), ("gray", 99), ("p99", 99), ("rgb", 73), ("gray", 58), ("p99", 53)]
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def prepare(args):
    import numpy as np
    import torch

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists():
        raise ValueError("Output already contains a prepared manifest; use it rather than silently rebuilding")
    snapshot = out / "trainer_snapshot" / "blurtrainer"
    snapshot.mkdir(parents=True, exist_ok=True)
    modules = ["__init__.py", "data.py", "synth.py", "fingerprint.py", "model.py"]
    for name in modules:
        shutil.copyfile(args.trainer_root / "blurtrainer" / name, snapshot / name)
    sys.path.insert(0, str(snapshot.parent))
    from blurtrainer.data import FingerprintDataset, read_index

    cfgs = {}
    for mode in ("rgb", "gray", "p99"):
        path = args.trainer_root / "runs" / f"pub_convnext_small_{mode}_synthphotos_frames_100ep_v1" / "config.json"
        cfgs[mode] = json.loads(path.read_text(encoding="utf-8"))
    cfg = cfgs["rgb"]
    index = args.trainer_root / cfg["index"]
    if digest(index) != EXPECTED_INDEX:
        raise ValueError("Frozen index hash mismatch")
    rows_all = read_index(index)
    rows = [row for row in rows_all if row["split"] == "test"]
    native = [row for row in rows if row["source"] == "frames"]
    photos = [row for row in rows if row["source"] == "photo"]
    groups = sorted({row["group"] for row in native})
    if (len(rows), len(native), len(photos), len(groups)) != (10592, 6592, 4000, 7):
        raise ValueError("Unexpected frozen test cohort")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test sample IDs")
    train_val_groups = {row["group"] for row in rows_all if row["source"] == "frames" and row["split"] != "test"}
    if train_val_groups.intersection(groups):
        raise ValueError("Native group crosses the held-out split")
    train_val_photos = {row["png"] for row in rows_all if row["source"] == "photo" and row["split"] != "test"}
    if train_val_photos.intersection(row["png"] for row in photos):
        raise ValueError("Photo original crosses the held-out split")
    index_copy = out / "frozen_index.csv"
    shutil.copyfile(index, index_copy)
    confidence = args.trainer_root / "data" / "confident_frames.csv"
    shutil.copyfile(confidence, out / "confident_frames.csv")
    synth_keys = ("synth_sharp_L", "synth_blur_L", "synth_mask_share", "frames_mask_share")
    synth_cfg = {key: cfg[key] for key in synth_keys}
    for other in cfgs.values():
        if {key: other[key] for key in synth_keys} != synth_cfg or other["dataset_root"] != cfg["dataset_root"]:
            raise ValueError("The checkpoints do not share the expected evaluation recipe")
    root = Path(cfg["dataset_root"])
    mask_folder = root / "frames" / "masks"
    masks = [{"path": str(path), "sha256": digest(path)} for path in sorted(mask_folder.glob("*.png"))]
    if not masks:
        raise ValueError("No source masks")
    sources = sorted({root / row["png"] for row in rows})
    print(json.dumps({"stage": "input_hashing", "unique_originals": len(sources)}), flush=True)
    def source_record(path):
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        source_records = list(pool.map(source_record, sources))
    checkpoints = []
    for mode, epoch in CHECKPOINTS:
        path = args.weights / f"{mode}_ep{epoch:03d}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        meta = ck["meta"]
        if meta["input"] != mode or meta["epoch"] != epoch:
            raise ValueError("Published checkpoint identity mismatch")
        checkpoints.append({"id": f"{mode}_ep{epoch:03d}", "input": mode, "epoch": epoch,
                            "weights_path": str(path.resolve()), "weights_sha256": digest(path),
                            "thresholds": {name: float(meta["thresholds"][name]) for name in ("r90", "r95", "r98")},
                            "selection_role": "primary" if epoch == 99 else "secondary"})
    ds = FingerprintDataset(rows, root, **synth_cfg)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    completed = 0
    records = []
    start = time.monotonic()
    def make_sample(i):
        sample = ds[(i, 0, False, False, None)]
        rgb, valid = sample["rgb"].numpy(), sample["valid"].numpy()
        path = cache / f"{i:05d}.npz"
        np.savez(path, rgb=rgb, valid=valid)
        return {"ordinal": i, "sample_id": rows[i]["sample_id"], "source": rows[i]["source"],
                "target": int(rows[i]["target"]), "group": rows[i]["group"],
                "cache_sha256": digest(path), "height": rgb.shape[1], "width": rgb.shape[2]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Executor.map preserves the frozen index order.
        for record in pool.map(make_sample, range(len(rows))):
            records.append(record)
            completed += 1
            if completed % 256 == 0 or completed == len(rows):
                atomic_json(out / "progress.json", {"stage": "prepare", "prepared": completed,
                                                    "total": len(rows), "elapsed_seconds": time.monotonic() - start})
                print(json.dumps({"stage": "prepare", "prepared": completed, "total": len(rows)}), flush=True)
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
    manifest = {"schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "index_sha256": digest(index_copy), "index_path": str(index_copy),
                "confidence_sha256": digest(out / "confident_frames.csv"),
                "checkpoints": checkpoints, "test_frames": len(native), "test_photos": len(photos),
                "test_groups": len(groups), "synth_config": synth_cfg,
                "source_hashes": {name: digest(snapshot / name) for name in modules},
                "runner_sha256": digest(__file__), "public_inference_sha256": digest(REPO / "blurdetect" / "model.py"),
                "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                             "ffmpeg": ffmpeg_version, "ffmpeg_sha256": digest(ffmpeg)},
                "private_source_files": source_records, "private_masks": masks, "private_cache": records,
                "mask_inventory_sha256": hashlib.sha256(json.dumps(masks, sort_keys=True).encode()).hexdigest(),
                "bootstrap_draws": 2000, "bootstrap_seed": 20261007,
                "prediction_parameters_locked": False,
                "access_history": "An earlier unguarded test script was already stalled at Windows worker spawning. "
                                  "It had read test metadata but produced no scores. Historical non-use is not independently established.",
                "threshold_source": "Published metadata, matching historical six-decimal validation-score quantiles; never recalibrated on test."}
    atomic_json(out / "run_manifest.json", manifest)
    print(json.dumps({"stage": "prepared", "samples": len(rows), "frames": len(native), "photos": len(photos)}), flush=True)


def run(args):
    import numpy as np
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, str(REPO))
    from blurdetect.model import load_detector
    out = args.out.resolve()
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(__file__) != manifest["runner_sha256"] or digest(REPO / "blurdetect" / "model.py") != manifest["public_inference_sha256"]:
        raise ValueError("Inference source changed after preparation")
    if digest(manifest["index_path"]) != manifest["index_sha256"]:
        raise ValueError("Prepared index changed")
    snapshot = out / "trainer_snapshot" / "blurtrainer"
    for name, sha in manifest["source_hashes"].items():
        if digest(snapshot / name) != sha:
            raise ValueError("Pinned trainer source changed")
    sys.path.insert(0, str(snapshot.parent))
    from blurtrainer.model import BlurDetector as TrainingDetector
    def load_training_detector(checkpoint):
        state = torch.load(checkpoint["weights_path"], map_location="cpu", weights_only=True)
        model = TrainingDetector("convnext_small", None, 4.0, False, checkpoint["input"])
        model.load_state_dict(state["state_dict"])
        return model.cuda().eval()
    records = manifest["private_cache"]
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    params = {"device": torch.cuda.get_device_name(), "precision": "CUDA bfloat16 autocast",
              "batch_size": args.batch, "memory_fraction": args.memory_fraction,
              "score_comparator": "blurred iff p >= stored validation threshold",
              "test_time_augmentation": False,
              "implementation": "Pinned trainer inference, matching the historical GPU-autocast path; public exported weights"}
    if manifest["prediction_parameters_locked"]:
        if manifest["prediction_parameters"] != params:
            raise ValueError("Resume parameters differ from the locked evaluation")
    else:
        # Probe only openly licensed demos; never test scores.
        from PIL import Image
        demo_paths = sorted((REPO / "test_images").rglob("*.png"))[:16]
        if not demo_paths:
            raise ValueError("Missing fixed openly licensed demo check")
        demo = np.stack([np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1) for path in demo_paths[:args.batch]])
        rgb = torch.from_numpy(demo).cuda()
        valid = torch.ones((len(demo), 1, *demo.shape[-2:]), dtype=torch.uint8, device="cuda")
        checks = []
        for checkpoint in manifest["checkpoints"][:3]:
            model, _ = load_detector(checkpoint["weights_path"], "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                public_scores = torch.sigmoid(model(rgb, valid)["image"].float()).cpu().numpy()
            del model
            torch.cuda.empty_cache()
            model = load_training_detector(checkpoint)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                training_scores = torch.sigmoid(model(rgb, valid)["image"].float()).cpu().numpy()
            if not np.isfinite(public_scores).all() or not np.isfinite(training_scores).all():
                raise ValueError("Non-finite demo check")
            checks.append({"input": checkpoint["input"], "max_public_vs_training_score_difference":
                           float(np.max(np.abs(public_scores - training_scores)))})
            # Memory preflight uses synthetic zeros at the largest cache geometry, never test scores.
            largest = max(records, key=lambda row: row["height"] * row["width"])
            dummy = torch.zeros((args.batch, 3, largest["height"], largest["width"]), dtype=torch.uint8, device="cuda")
            dummy_valid = torch.ones_like(dummy[:, :1])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                dummy_out = model(dummy, dummy_valid)["image"]
            if not bool(torch.isfinite(dummy_out).all()):
                raise ValueError("Non-finite memory preflight")
            del model, dummy, dummy_valid, dummy_out
            torch.cuda.empty_cache()
        manifest["demo_check"] = {"samples_per_input": len(demo), "finite": True, "comparisons": checks,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "demo_file_sha256": [digest(path) for path in demo_paths[:args.batch]]}
        del rgb, valid
        torch.cuda.empty_cache()
        manifest["prediction_parameters"] = params
        manifest["prediction_parameters_locked"] = True
        manifest["locked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
    score_dir = out / "scores"
    score_dir.mkdir(exist_ok=True)
    by_shape = {}
    for record in records:
        by_shape.setdefault((record["height"], record["width"]), []).append(record)
    start = time.monotonic()
    manifest_hash = digest(manifest_path)
    def load_sample(record):
        path = out / "cache" / f'{record["ordinal"]:05d}.npz'
        if digest(path) != record["cache_sha256"]:
            raise ValueError("Prepared sample bytes changed")
        with np.load(path, allow_pickle=False) as sample:
            return sample["rgb"], sample["valid"]
    for checkpoint in manifest["checkpoints"]:
        if digest(checkpoint["weights_path"]) != checkpoint["weights_sha256"]:
            raise ValueError("Checkpoint bytes changed")
        csv_path = score_dir / f'{checkpoint["id"]}.csv'
        sidecar = score_dir / f'{checkpoint["id"]}.json'
        if sidecar.exists():
            old = json.loads(sidecar.read_text())
            if old["manifest_sha256"] != manifest_hash:
                raise ValueError("Score cache belongs to a different manifest")
        elif csv_path.exists():
            raise ValueError("Unbound score cache")
        done = {}
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["sample_id"] in done:
                        raise ValueError("Duplicate saved score")
                    value = float(row["p_blurred"])
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid saved score")
                    done[row["sample_id"]] = row
            known = {row["sample_id"]: row for row in records}
            if any(key not in known or int(row["target"]) != known[key]["target"] or row["source"] != known[key]["source"]
                   for key, row in done.items()):
                raise ValueError("Saved score cohort mismatch")
        else:
            atomic_json(sidecar, {"manifest_sha256": manifest_hash, "complete": False})
        if len(done) == len(records):
            print(json.dumps({"checkpoint": checkpoint["id"], "already_complete": True}), flush=True)
            continue
        model = load_training_detector(checkpoint)
        with csv_path.open("a", newline="", encoding="utf-8") as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            writer = csv.writer(f)
            if not done:
                writer.writerow(["sample_id", "source", "target", "p_blurred"])
            for shape, group in by_shape.items():
                for offset in range(0, len(group), args.batch):
                    batch = group[offset:offset + args.batch]
                    # Preserve original batch membership on resume; only missing scores are written.
                    if all(record["sample_id"] in done for record in batch):
                        continue
                    arrays = list(pool.map(load_sample, batch))
                    rgb = torch.from_numpy(np.stack([row[0] for row in arrays])).cuda()
                    valid = torch.from_numpy(np.stack([row[1] for row in arrays])).cuda()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        p = torch.sigmoid(model(rgb, valid)["image"].float()).cpu().numpy()
                    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                        raise ValueError("Non-finite test predictions")
                    for record, score in zip(batch, p):
                        if record["sample_id"] not in done:
                            writer.writerow([record["sample_id"], record["source"], record["target"], format(float(score), ".17g")])
                            done[record["sample_id"]] = True
                    f.flush()
                    del rgb, valid
                    if len(done) % 128 < args.batch or len(done) == len(records):
                        progress = {"stage": "inference", "checkpoint": checkpoint["id"], "scored": len(done),
                                    "total": len(records), "elapsed_seconds": time.monotonic() - start,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                        atomic_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
        if len(done) != len(records):
            raise ValueError("Incomplete checkpoint predictions")
        atomic_json(sidecar, {"manifest_sha256": manifest_hash, "complete": True, "scores_sha256": digest(csv_path),
                              "samples": len(done), "weights_sha256": checkpoint["weights_sha256"]})
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"stage": "checkpoint_complete", "checkpoint": checkpoint["id"], "samples": len(done)}), flush=True)
    atomic_json(out / "progress.json", {"stage": "complete", "checkpoints": len(manifest["checkpoints"]),
                                        "scores": len(records) * len(manifest["checkpoints"]),
                                        "elapsed_seconds": time.monotonic() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--trainer-root", type=Path, required=True)
    prep.add_argument("--weights", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=4)
    score = commands.add_parser("run")
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--batch", type=int, default=8)
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--memory-fraction", type=float, default=0.09)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("At least one input worker is required")
    if args.command == "prepare":
        prepare(args)
    else:
        if not 1 <= args.batch <= 8 or not 0 < args.memory_fraction <= 0.85:
            parser.error("Invalid bounded inference parameters")
        run(args)


if __name__ == "__main__":
    main()
