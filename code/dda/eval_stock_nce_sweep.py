#!/usr/bin/env python3
"""Select the stock AlphaPeptDeep prediction NCE on the frozen validation fold.

The final public test and external DDA data are never inspected by this script.
By default, validation sources whose basename starts with ``P_putida`` are
excluded so the external P. putida runs remain untouched by NCE selection.
Sampling exactly mirrors ``t3_finetune.load_fold``: one shared ``random.Random``
instance, file-list order preserved, and a uniform per-file cap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_paired_full as paired
from eval_bacterial_hcd import load_species_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val_fold_list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--nce_grid", default="5,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40"
    )
    parser.add_argument("--exclude_source_prefix", default="P_putida")
    parser.add_argument("--max_per_file", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction_batch_size", type=int, default=2000)
    args = parser.parse_args()
    args.nce_values = [float(item) for item in args.nce_grid.split(",") if item.strip()]
    if not args.nce_values or any(not 0 < value <= 100 for value in args.nce_values):
        parser.error("--nce_grid must contain values in (0, 100]")
    if len(set(args.nce_values)) != len(args.nce_values):
        parser.error("--nce_grid contains duplicate values")
    if args.max_per_file < 1 or args.prediction_batch_size < 1:
        parser.error("caps and batch sizes must be positive")
    return args


def load_validation(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[dict]]:
    list_path = Path(args.val_fold_list).resolve()
    paths = [Path(line.strip()) for line in list_path.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    selected: dict[str, list[dict]] = {}
    manifest: list[dict] = []
    global_index = 0
    for path in paths:
        excluded = path.stem.startswith(args.exclude_source_prefix)
        record = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "excluded_by_prefix": excluded,
        }
        rows = load_species_file(path, 10**12)
        record["eligible_rows"] = len(rows)
        if len(rows) > args.max_per_file:
            rows = rng.sample(rows, args.max_per_file)
        # Consume the RNG exactly as the original full-fold loader did, then
        # remove P. putida as a strict row subset. Skipping the file before the
        # sample call would change every later file's sampled spectra.
        if excluded:
            record["sampled_before_exclusion"] = len(rows)
            record["selected_rows"] = 0
            manifest.append(record)
            del rows
            continue
        for local_index, row in enumerate(rows):
            row["spectrum_id"] = f"validation:{global_index}"
            row["scan"] = local_index
            row["selection_rank"] = local_index
            global_index += 1
        record["selected_rows"] = len(rows)
        manifest.append(record)
        selected[path.stem] = rows
        print(f"[{path.stem}] selected={len(rows)}", flush=True)
    if not selected:
        raise RuntimeError("No validation sources remained after exclusion")
    return selected, manifest


def finite_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def summarize(results: dict[str, dict]) -> dict:
    ok = [record for record in results.values() if record["status"] == "ok"]
    return {
        "n_predictions": len(results),
        "n_valid": len(ok),
        "status_counts": dict(sorted(Counter(
            record["status"] for record in results.values()
        ).items())),
        "spearman_union_top7_mean": finite_mean(
            [record["spearman_union_top7"] for record in ok]
        ),
        "spearman_obs_top7_mean": finite_mean(
            [record["spearman_obs_top7"] for record in ok]
        ),
        "spectral_angle_mean": finite_mean([record["spectral_angle"] for record in ok]),
        "cosine_mean": finite_mean([record["cosine"] for record in ok]),
        "pcc_mean": finite_mean([record["pcc"] for record in ok]),
        "pcc90": finite_mean([record["pcc90"] for record in ok]),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"ERROR: output already exists: {out}")
    selected, manifest = load_validation(args)
    expected = sum(item["selected_rows"] for item in manifest)
    print(f"Validation spectra after exclusion: {expected}", flush=True)

    from peptdeep.pretrained_models import ModelManager

    manager = ModelManager()
    manager.load_installed_models()
    results_by_nce: dict[str, dict] = {}
    expected_valid = None
    for nce in args.nce_values:
        print(f"\n=== stock prediction NCE {nce:g} ===", flush=True)
        for rows in selected.values():
            for row in rows:
                row["nce"] = nce
        predictions = paired.evaluate_model(
            manager,
            selected,
            topn=7,
            prediction_batch_size=args.prediction_batch_size,
            label=f"stock_nce_{nce:g}",
        )
        summary = summarize(predictions)
        if summary["n_predictions"] != expected:
            raise RuntimeError(f"NCE {nce:g}: prediction denominator drift: {summary}")
        if expected_valid is None:
            expected_valid = summary["n_valid"]
            if expected_valid < expected - 10:
                raise RuntimeError(f"Unexpectedly many invalid validation rows: {summary}")
        elif summary["n_valid"] != expected_valid:
            raise RuntimeError(
                f"NCE {nce:g}: valid denominator changed from {expected_valid}: {summary}"
            )
        results_by_nce[f"{nce:g}"] = summary
        print(json.dumps(summary, indent=2), flush=True)
        del predictions
        gc.collect()
        paired.release_gpu_cache()

    selected_nce = min(
        args.nce_values,
        key=lambda value: (
            -results_by_nce[f"{value:g}"]["spearman_union_top7_mean"],
            abs(value - 30.0),
            value,
        ),
    )
    boundary_selected = selected_nce in (min(args.nce_values), max(args.nce_values))
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": (
            "boundary_requires_extension" if boundary_selected else "frozen_validation_only"
        ),
        "selection_metric": "mean union-top-7 Spearman",
        "tie_break": "closest to nominal NCE 30, then lower NCE",
        "selected_stock_prediction_nce": selected_nce,
        "boundary_selected": boundary_selected,
        "candidate_nce_values": args.nce_values,
        "excluded_source_prefix": args.exclude_source_prefix,
        "n_validation_spectra": expected,
        "n_valid_per_candidate": expected_valid,
        "max_per_file": args.max_per_file,
        "seed": args.seed,
        "instrument": "QE",
        "results": results_by_nce,
        "validation_manifest": manifest,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "val_fold_list": str(Path(args.val_fold_list).resolve()),
            "val_fold_list_sha256": sha256_file(Path(args.val_fold_list).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    try:
        import peptdeep

        payload["provenance"]["peptdeep_version"] = getattr(peptdeep, "__version__", "unknown")
    except Exception as exc:
        payload["provenance"]["peptdeep_version"] = f"unavailable: {exc}"
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(out)
    print(f"\nSELECTED STOCK PREDICTION NCE: {selected_nce:g}", flush=True)
    print(f"Wrote {out}", flush=True)
    if boundary_selected:
        raise SystemExit(
            "ERROR: the validation optimum is on the NCE grid boundary; extend the grid before "
            "freezing or running test data"
        )


if __name__ == "__main__":
    main()
