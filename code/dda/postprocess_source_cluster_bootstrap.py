#!/usr/bin/env python3
"""Add source-stratified modified-precursor cluster uncertainty to sensitivity output."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from source_cluster_bootstrap_stats import source_stratified_cluster_bootstrap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity_dir", required=True)
    parser.add_argument("--primary_summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260818)
    parser.add_argument("--bootstrap_block", type=int, default=32)
    return parser.parse_args()


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return float(np.mean(values[np.isfinite(values)]))


def main():
    args = parse_args()
    sensitivity_dir = Path(args.sensitivity_dir).resolve()
    tsv = sensitivity_dir / "peakmatch_sensitivity_per_spectrum.tsv"
    summary_path = sensitivity_dir / "peakmatch_sensitivity_summary.json"
    primary_path = Path(args.primary_summary).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"ERROR: output already exists: {out}")
    frame = pd.read_csv(tsv, sep="\t", low_memory=False)
    sensitivity = json.loads(summary_path.read_text())
    primary = json.loads(primary_path.read_text())
    labels = sorted(sensitivity["scoring_configurations"])

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "source-stratified modified-precursor cluster bootstrap",
        "dataset": sensitivity["dataset"],
        "inputs": {
            "sensitivity_tsv": str(tsv),
            "sensitivity_tsv_sha256": sha256(tsv),
            "sensitivity_summary": str(summary_path),
            "sensitivity_summary_sha256": sha256(summary_path),
            "primary_summary": str(primary_path),
            "primary_summary_sha256": sha256(primary_path),
        },
        "configurations": {},
        "historical_regression_check": {},
    }
    for config_index, label in enumerate(labels):
        valid = frame.loc[frame[f"{label}_valid_pair"] == True].copy()  # noqa: E712
        config = {"n_valid_pair": int(len(valid)), "metrics": {}}
        for metric_index, metric in enumerate(("spearman_union_top7", "pcc90")):
            scale = 100.0 if metric == "pcc90" else 1.0
            config["metrics"][metric] = source_stratified_cluster_bootstrap(
                valid,
                f"{label}_delta_{metric}",
                "source",
                "modified_precursor_cluster",
                args.bootstrap_reps,
                args.bootstrap_seed + config_index * 100 + metric_index,
                args.bootstrap_block,
                scale,
            )
            if metric == "pcc90":
                config["metrics"][metric]["units"] = "percentage points"
        output["configurations"][label] = config

    historical = "nearest_0p5Da"
    current = sensitivity["scoring_configurations"][historical]
    checks = {}
    for model_name, primary_name in (("stock", "stock"), ("fine_tuned", "fine_tuned")):
        for metric in (
            "spearman_union_top7",
            "spectral_angle",
            "cosine",
            "pcc",
            "pcc90",
        ):
            current_value = current[model_name][f"{metric}_mean"]
            primary_key = f"{metric}_mean" if metric != "pcc90" else "pcc90"
            primary_value = primary["overall"][primary_name][primary_key]
            checks[f"{model_name}_{metric}"] = {
                "sensitivity": current_value,
                "primary": primary_value,
                "absolute_difference": abs(current_value - primary_value),
            }
    output["historical_regression_check"] = {
        "configuration": historical,
        "checks": checks,
        "maximum_absolute_difference": max(
            item["absolute_difference"] for item in checks.values()
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temp.replace(out)
    print(json.dumps(output["historical_regression_check"], indent=2))


if __name__ == "__main__":
    main()
