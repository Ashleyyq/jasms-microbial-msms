#!/usr/bin/env python3
"""Stratify the external DDA result by exact fine-tuning exposure.

The actual 360,000-spectrum training sample is reconstructed with the frozen
file list, clean-sequence set, per-file cap, and seed. External spectra are
classified at two levels: bare peptide sequence and modified precursor
(sequence, modifications, sites, charge). No model prediction is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_bacterial_hcd import load_species_file
from source_cluster_bootstrap_stats import source_stratified_cluster_bootstrap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fold_list", required=True)
    parser.add_argument("--clean_sequences", required=True)
    parser.add_argument("--paired_tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_per_file", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260819)
    parser.add_argument("--bootstrap_block", type=int, default=32)
    return parser.parse_args()


def precursor_key(sequence: str, mods: str, sites: str, charge: int) -> str:
    return f"{sequence}|{mods}|{sites}|z{int(charge)}"


def reconstruct_sample(args: argparse.Namespace) -> tuple[set[str], set[str], int, list[dict]]:
    clean_path = Path(args.clean_sequences).resolve()
    clean = {line.strip() for line in clean_path.read_text().splitlines() if line.strip()}
    paths = [
        Path(line.strip())
        for line in Path(args.train_fold_list).resolve().read_text().splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    sampled_sequences: set[str] = set()
    sampled_precursors: set[str] = set()
    n_sampled = 0
    manifest: list[dict] = []
    for path in paths:
        rows = load_species_file(path, 10**12)
        eligible_before_clean = len(rows)
        rows = [row for row in rows if row["sequence"] in clean]
        eligible_after_clean = len(rows)
        if len(rows) > args.max_per_file:
            rows = rng.sample(rows, args.max_per_file)
        for row in rows:
            sampled_sequences.add(row["sequence"])
            sampled_precursors.add(
                precursor_key(
                    row["sequence"], row["mods"], row["mod_sites"], row["charge"]
                )
            )
        n_sampled += len(rows)
        manifest.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "eligible_before_clean_filter": eligible_before_clean,
                "eligible_after_clean_filter": eligible_after_clean,
                "sampled_rows": len(rows),
            }
        )
        print(f"[{path.stem}] sampled={len(rows)}", flush=True)
        del rows
    return sampled_sequences, sampled_precursors, n_sampled, manifest


def finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def summarize_group(frame: pd.DataFrame, args: argparse.Namespace, seed_offset: int) -> dict:
    valid = frame.loc[frame["valid_pair"]].copy()
    result = {
        "n_spectra": int(len(frame)),
        "n_valid_pair": int(len(valid)),
        "n_unique_sequences": int(frame["sequence"].nunique()),
        "n_modified_precursors": int(frame["modified_precursor_cluster"].nunique()),
        "stock_spearman_union_top7_mean": finite_mean(valid["stock_spearman_union_top7"]),
        "fine_spearman_union_top7_mean": finite_mean(valid["fine_spearman_union_top7"]),
        "delta_spearman_union_top7_mean": finite_mean(valid["delta_spearman_union_top7"]),
        "stock_pcc90": finite_mean(valid["stock_pcc90"]),
        "fine_pcc90": finite_mean(valid["fine_pcc90"]),
        "delta_pcc90_percentage_points": (
            100.0 * finite_mean(valid["delta_pcc90"]) if len(valid) else None
        ),
        "per_source": {},
        "cluster_bootstrap": {},
    }
    for source, source_frame in valid.groupby("source", sort=True):
        result["per_source"][str(source)] = {
            "n_valid_pair": int(len(source_frame)),
            "delta_spearman_union_top7_mean": finite_mean(
                source_frame["delta_spearman_union_top7"]
            ),
            "delta_pcc90_percentage_points": 100.0 * finite_mean(
                source_frame["delta_pcc90"]
            ),
        }
    if len(valid):
        for metric_index, (metric, scale) in enumerate(
            (("delta_spearman_union_top7", 1.0), ("delta_pcc90", 100.0))
        ):
            result["cluster_bootstrap"][metric] = source_stratified_cluster_bootstrap(
                valid,
                metric,
                "source",
                "modified_precursor_cluster",
                args.bootstrap_reps,
                args.bootstrap_seed + seed_offset * 10 + metric_index,
                args.bootstrap_block,
                scale,
            )
    return result


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"ERROR: output already exists: {out}")
    sampled_sequences, sampled_precursors, n_sampled, train_manifest = reconstruct_sample(args)
    if n_sampled != 360000:
        raise RuntimeError(f"Training-sample denominator drift: {n_sampled}, expected 360000")

    clean = {
        line.strip()
        for line in Path(args.clean_sequences).resolve().read_text().splitlines()
        if line.strip()
    }
    paired_path = Path(args.paired_tsv).resolve()
    frame = pd.read_csv(paired_path, sep="\t", low_memory=False)
    if len(frame) != 73529:
        raise RuntimeError(f"External paired denominator drift: {len(frame)}, expected 73529")
    frame["valid_pair"] = frame["valid_pair"].astype(str).str.lower().eq("true")
    for column in ("mods", "mod_sites"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["modified_precursor_cluster"] = [
        precursor_key(seq, mods, sites, charge)
        for seq, mods, sites, charge in zip(
            frame["sequence"], frame["mods"], frame["mod_sites"], frame["charge"]
        )
    ]
    frame["sequence_exposure"] = np.where(
        frame["sequence"].isin(sampled_sequences),
        "seen_in_actual_finetune_sample",
        np.where(
            frame["sequence"].isin(clean),
            "available_in_train_fold_but_not_sampled",
            "absent_from_clean_train_fold",
        ),
    )
    frame["modified_precursor_exposure"] = np.where(
        frame["modified_precursor_cluster"].isin(sampled_precursors),
        "seen_in_actual_finetune_sample",
        "not_seen_as_modified_precursor",
    )

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "external DDA exact-exposure stratification",
        "definitions": {
            "sequence": "exact unmodified amino-acid sequence",
            "modified_precursor": "sequence + modification names + modification sites + charge",
            "actual_finetune_sample": (
                "deterministic reconstruction using the frozen 180-file train fold, clean sequence "
                "filter, cap 2000 per file, and random seed 42"
            ),
        },
        "training_reconstruction": {
            "n_sampled_spectra": n_sampled,
            "n_unique_sampled_sequences": len(sampled_sequences),
            "n_unique_sampled_modified_precursors": len(sampled_precursors),
            "n_clean_train_fold_sequences": len(clean),
            "manifest": train_manifest,
        },
        "external_denominator": int(len(frame)),
        "sequence_groups": {},
        "modified_precursor_groups": {},
        "inputs": {
            "train_fold_list": str(Path(args.train_fold_list).resolve()),
            "train_fold_list_sha256": sha256_file(Path(args.train_fold_list).resolve()),
            "clean_sequences": str(Path(args.clean_sequences).resolve()),
            "clean_sequences_sha256": sha256_file(Path(args.clean_sequences).resolve()),
            "paired_tsv": str(paired_path),
            "paired_tsv_sha256": sha256_file(paired_path),
        },
    }
    for index, (label, group) in enumerate(frame.groupby("sequence_exposure", sort=True)):
        output["sequence_groups"][str(label)] = summarize_group(group, args, index)
    for index, (label, group) in enumerate(
        frame.groupby("modified_precursor_exposure", sort=True), start=100
    ):
        output["modified_precursor_groups"][str(label)] = summarize_group(group, args, index)

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temp.replace(out)
    print(json.dumps({
        "sequence_groups": {
            key: {"n_spectra": value["n_spectra"],
                  "delta_spearman": value["delta_spearman_union_top7_mean"]}
            for key, value in output["sequence_groups"].items()
        },
        "modified_precursor_groups": {
            key: {"n_spectra": value["n_spectra"],
                  "delta_spearman": value["delta_spearman_union_top7_mean"]}
            for key, value in output["modified_precursor_groups"].items()
        },
    }, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
