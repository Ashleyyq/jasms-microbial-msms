#!/usr/bin/env python3
"""Versioned APD peak-matching and clustered-uncertainty sensitivity run.

This analysis does not overwrite or reinterpret the archived primary outputs. It
predicts the same spectra with stock and frozen fine-tuned APD models, then scores
each prediction under four ground-truth constructions:

* nearest experimental peak within 0.5 Da (historical evaluator);
* highest-intensity experimental peak within 0.5 Da (training helper convention);
* nearest experimental peak within a stricter configurable Da tolerance; and
* highest-intensity experimental peak within that stricter tolerance.

It also resamples whole modified-sequence+charge clusters, preserving all spectra
belonging to a sampled precursor cluster.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_paired_full as base
from peakmatch_sensitivity_stats import cluster_bootstrap, match_ground_truth_pair


METRICS = (
    "spearman_union_top7",
    "spearman_obs_top7",
    "spectral_angle",
    "cosine",
    "pcc",
    "pcc90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_label", required=True)
    parser.add_argument("--finetune_ckpt", required=True)
    parser.add_argument("--checkpoint_label", required=True)
    parser.add_argument("--max_spectra", type=int, required=True)
    parser.add_argument("--test_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topn", type=int, default=7)
    parser.add_argument("--prediction_batch_size", type=int, default=1000)
    parser.add_argument("--historical_tolerance_da", type=float, default=0.5)
    parser.add_argument("--strict_tolerance_da", type=float, default=0.02)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260817)
    parser.add_argument("--bootstrap_block", type=int, default=32)
    args = parser.parse_args()
    if not 0 < args.test_ratio <= 1:
        parser.error("--test_ratio must be in (0, 1]")
    if args.max_spectra < 0 or args.prediction_batch_size < 1:
        parser.error("invalid parser cap or prediction batch size")
    if args.topn != 7:
        parser.error("manuscript metric is fixed at topn=7")
    if not 0 < args.strict_tolerance_da < args.historical_tolerance_da:
        parser.error("strict tolerance must be positive and below historical tolerance")
    if args.bootstrap_reps < 1 or args.bootstrap_block < 1:
        parser.error("bootstrap counts must be positive")
    return args


def prediction_vectors(manager, rows: list[dict], batch_size: int, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        precursor_df, fragment_df = base.predict_species(manager, batch)
        if len(precursor_df) != len(batch):
            raise RuntimeError(f"{label} precursor row mismatch")
        for index, row in enumerate(batch):
            precursor = precursor_df.iloc[index]
            start = int(precursor["frag_start_idx"])
            stop = int(precursor["frag_stop_idx"])
            n_ions = stop - start
            if n_ions <= 0:
                output[row["spectrum_id"]] = {"status": "no_predicted_fragment_rows"}
                continue
            pred_b = fragment_df.iloc[start:stop]["b_z1"].to_numpy(dtype=float)[:n_ions]
            pred_y = fragment_df.iloc[start:stop]["y_z1"].to_numpy(dtype=float)[:n_ions]
            pred = np.concatenate([pred_b, pred_y[::-1]])
            if not len(pred) or not np.isfinite(pred).all() or np.max(pred) <= 0:
                output[row["spectrum_id"]] = {"status": "invalid_prediction"}
                continue
            output[row["spectrum_id"]] = {
                "status": "ok",
                "n_ions": int(n_ions),
                "vector": pred / np.max(pred),
            }
        print(f"[{label}] {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
        del precursor_df, fragment_df, batch
        gc.collect()
    return output


def vector_metrics(pred: np.ndarray, gt: np.ndarray, topn: int) -> dict:
    if len(pred) != len(gt):
        return {"status": "length_mismatch"}
    if not len(gt) or not np.isfinite(gt).all() or np.max(gt) <= 0:
        return {"status": "zero_or_invalid_ground_truth"}
    pcc = base.pearson_corr(pred, gt)
    union, union_size = base.spearman_union_topN(gt, pred, topn)
    return {
        "status": "ok",
        "spearman_union_top7": float(union),
        "spearman_obs_top7": float(base.spearman_obs_topN(gt, pred, topn)),
        "spectral_angle": float(base.angular_similarity(pred, gt)),
        "cosine": float(base.cosine(pred, gt)),
        "pcc": float(pcc),
        "pcc90": float(pcc >= 0.9),
        "union_size": int(union_size),
    }


def config_label(strategy: str, tolerance: float, historical: float) -> str:
    suffix = "0p5Da" if np.isclose(tolerance, historical) else f"{tolerance:g}Da".replace(".", "p")
    return f"{strategy}_{suffix}"


def build_frame(
    rows: list[dict],
    stock: dict[str, dict],
    fine: dict[str, dict],
    historical_tolerance: float,
    strict_tolerance: float,
    topn: int,
) -> tuple[pd.DataFrame, list[str]]:
    labels = [
        config_label("nearest", historical_tolerance, historical_tolerance),
        config_label("highest", historical_tolerance, historical_tolerance),
        config_label("nearest", strict_tolerance, historical_tolerance),
        config_label("highest", strict_tolerance, historical_tolerance),
    ]
    records = []
    for row in rows:
        sid = row["spectrum_id"]
        stock_pred, fine_pred = stock[sid], fine[sid]
        if stock_pred["status"] != "ok" or fine_pred["status"] != "ok":
            raise RuntimeError(f"Prediction failure for {sid}: {stock_pred}, {fine_pred}")
        if stock_pred["n_ions"] != fine_pred["n_ions"]:
            raise RuntimeError(f"Stock/fine ion-count mismatch for {sid}")
        n_ions = stock_pred["n_ions"]
        b_mz, y_mz = base.compute_fragment_mz(row["sequence"], row["mods"], row["mod_sites"])
        b_mz, y_mz = b_mz[:n_ions], y_mz[:n_ions]

        rec = {
            "spectrum_id": sid,
            "source": row["source"],
            "mgf_entry_ordinal": row["scan"],
            "sequence": row["sequence"],
            "mods": row["mods"],
            "mod_sites": row["mod_sites"],
            "charge": row["charge"],
            "modified_precursor_cluster": (
                f"{row['sequence']}|{row['mods']}|{row['mod_sites']}|z{row['charge']}"
            ),
        }
        gt_by_label: dict[str, np.ndarray] = {}
        for tolerance, tolerance_name in (
            (historical_tolerance, "0p5Da"),
            (strict_tolerance, f"{strict_tolerance:g}Da".replace(".", "p")),
        ):
            nearest, highest, diag = match_ground_truth_pair(
                row["exp_mz"], row["exp_intensity"], b_mz, y_mz, tolerance
            )
            gt_by_label[config_label("nearest", tolerance, historical_tolerance)] = nearest
            gt_by_label[config_label("highest", tolerance, historical_tolerance)] = highest
            for key, value in diag.items():
                rec[f"match_{tolerance_name}_{key}"] = value

        for label in labels:
            for model_name, prediction in (("stock", stock_pred), ("fine", fine_pred)):
                metrics = vector_metrics(prediction["vector"], gt_by_label[label], topn)
                for key, value in metrics.items():
                    rec[f"{label}_{model_name}_{key}"] = value
            rec[f"{label}_valid_pair"] = (
                rec[f"{label}_stock_status"] == "ok"
                and rec[f"{label}_fine_status"] == "ok"
            )
            for metric in METRICS:
                stock_value = rec.get(f"{label}_stock_{metric}", np.nan)
                fine_value = rec.get(f"{label}_fine_{metric}", np.nan)
                rec[f"{label}_delta_{metric}"] = fine_value - stock_value
        records.append(rec)
    return pd.DataFrame.from_records(records), labels


def finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def summarize_config(
    frame: pd.DataFrame,
    label: str,
    reps: int,
    seed: int,
    block: int,
) -> dict:
    valid = frame.loc[frame[f"{label}_valid_pair"]].copy()
    result = {
        "n_valid_pair": int(len(valid)),
        "n_modified_precursor_clusters": int(valid["modified_precursor_cluster"].nunique()),
        "stock": {},
        "fine_tuned": {},
        "difference": {},
        "per_source": {},
    }
    for model_name, output_name in (("stock", "stock"), ("fine", "fine_tuned")):
        for metric in METRICS:
            column = f"{label}_{model_name}_{metric}"
            result[output_name][f"{metric}_mean"] = finite_mean(valid[column])
            result[output_name][f"n_{metric}"] = int(
                np.isfinite(pd.to_numeric(valid[column], errors="coerce")).sum()
            )

    for metric_index, metric in enumerate(("spearman_union_top7", "pcc90")):
        scale = 100.0 if metric == "pcc90" else 1.0
        result["difference"][metric] = cluster_bootstrap(
            valid,
            f"{label}_delta_{metric}",
            "modified_precursor_cluster",
            reps,
            seed + metric_index,
            block,
            scale,
        )
        if metric == "pcc90":
            result["difference"][metric]["units"] = "percentage points"

    for source, group in valid.groupby("source", sort=True):
        result["per_source"][source] = {
            "n": int(len(group)),
            "n_modified_precursor_clusters": int(group["modified_precursor_cluster"].nunique()),
            "stock_spearman_union_top7_mean": finite_mean(
                group[f"{label}_stock_spearman_union_top7"]
            ),
            "fine_spearman_union_top7_mean": finite_mean(
                group[f"{label}_fine_spearman_union_top7"]
            ),
            "delta_spearman_union_top7_mean": finite_mean(
                group[f"{label}_delta_spearman_union_top7"]
            ),
            "delta_pcc90_percentage_points": (
                finite_mean(group[f"{label}_delta_pcc90"]) * 100.0
            ),
        }
    return result


def summarize_matching(frame: pd.DataFrame, tolerance_name: str) -> dict:
    targets = int(frame[f"match_{tolerance_name}_n_targets"].sum())
    matched = int(frame[f"match_{tolerance_name}_n_targets_with_match"].sum())
    multihit = int(frame[f"match_{tolerance_name}_n_multihit_targets"].sum())
    changed = int(frame[f"match_{tolerance_name}_n_nearest_highest_different"].sum())
    return {
        "n_theoretical_targets": targets,
        "n_targets_with_at_least_one_peak": matched,
        "n_targets_with_multiple_candidate_peaks": multihit,
        "fraction_multihit_all_targets": multihit / targets if targets else None,
        "fraction_multihit_matched_targets": multihit / matched if matched else None,
        "n_targets_changed_by_nearest_vs_highest": changed,
        "fraction_changed_all_targets": changed / targets if targets else None,
        "fraction_changed_matched_targets": changed / matched if matched else None,
        "max_per_target_absolute_intensity_difference": finite_mean(
            frame[f"match_{tolerance_name}_max_abs_difference"]
        ),
        "note": "The final field is the mean across spectra of each spectrum's maximum difference.",
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    checkpoint = Path(os.path.expanduser(args.finetune_ckpt)).resolve()
    mgf_files = sorted(data_dir.glob("*.mgf"))
    if not data_dir.is_dir() or not checkpoint.is_file() or not mgf_files:
        raise SystemExit("ERROR: missing input directory, checkpoint, or MGF files")
    if out_dir.exists():
        raise SystemExit(f"ERROR: output path already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    shard_dir = out_dir / "sensitivity_shards"
    shard_dir.mkdir()

    from peptdeep.pretrained_models import ModelManager

    stock_manager = ModelManager()
    stock_manager.load_installed_models()
    fine_manager = ModelManager()
    fine_manager.load_installed_models()
    base._load_finetuned_ms2(fine_manager, str(checkpoint))

    frames = []
    manifests = []
    labels = None
    for file_index, mgf in enumerate(mgf_files):
        rows, counts = base.load_file_with_ids(mgf, args.max_spectra)
        # Preserve the historical evaluator's legacy NumPy RNG path. At ratio
        # 1.0 this only changes processing order, but matching the old path also
        # makes the historical 0.5-Da configuration an exact regression check.
        np.random.seed(args.seed + file_index)
        permutation = np.random.permutation(len(rows))
        n_selected = max(1, int(len(rows) * args.test_ratio))
        chosen = [rows[int(i)] for i in permutation[:n_selected]]
        counts["selected_rows"] = len(chosen)
        counts["selection_seed"] = args.seed + file_index
        manifests.append(counts)
        print(f"[{mgf.stem}] predicting {len(chosen)} rows", flush=True)
        stock = prediction_vectors(stock_manager, chosen, args.prediction_batch_size, "stock")
        fine = prediction_vectors(fine_manager, chosen, args.prediction_batch_size, "fine")
        frame, current_labels = build_frame(
            chosen,
            stock,
            fine,
            args.historical_tolerance_da,
            args.strict_tolerance_da,
            args.topn,
        )
        labels = labels or current_labels
        if labels != current_labels:
            raise RuntimeError("configuration labels changed between files")
        shard = shard_dir / f"{file_index:02d}_{mgf.stem}.tsv"
        temp = shard.with_suffix(".tsv.tmp")
        frame.to_csv(temp, sep="\t", index=False, na_rep="NA")
        temp.replace(shard)
        frames.append(frame)
        del rows, chosen, stock, fine, frame
        gc.collect()

    del stock_manager, fine_manager
    base.release_gpu_cache()
    combined = pd.concat(frames, ignore_index=True)
    output_tsv = out_dir / "peakmatch_sensitivity_per_spectrum.tsv"
    temp_tsv = output_tsv.with_suffix(".tsv.tmp")
    combined.to_csv(temp_tsv, sep="\t", index=False, na_rep="NA")
    temp_tsv.replace(output_tsv)

    strict_name = f"{args.strict_tolerance_da:g}Da".replace(".", "p")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "peak matching and modified-precursor cluster sensitivity",
        "dataset": args.dataset_label,
        "checkpoint_label": args.checkpoint_label,
        "config": vars(args),
        "denominators": {
            "mgf_entries": int(sum(x["mgf_entries"] for x in manifests)),
            "parser_records": int(sum(x["parser_records"] for x in manifests)),
            "eligible_rows": int(sum(x["eligible_rows"] for x in manifests)),
            "selected_rows": int(len(combined)),
            "unique_modified_precursor_clusters": int(
                combined["modified_precursor_cluster"].nunique()
            ),
        },
        "matching_diagnostics": {
            "0p5Da": summarize_matching(combined, "0p5Da"),
            strict_name: summarize_matching(combined, strict_name),
        },
        "scoring_configurations": {},
        "input_files": manifests,
    }
    for index, label in enumerate(labels or []):
        summary["scoring_configurations"][label] = summarize_config(
            combined,
            label,
            args.bootstrap_reps,
            args.bootstrap_seed + index * 100,
            args.bootstrap_block,
        )

    dependency_paths = {}
    for name, function in {
        "eval_paired_full": base.load_file_with_ids,
        "run_all": base.parse_mgf_file,
        "eval_koina_bacterial": base.compute_fragment_mz,
        "eval_local_zeroshot_baseline": base.predict_species,
        "rescore_spearman_union": base.spearman_union_topN,
        "peakmatch_sensitivity_stats": match_ground_truth_pair,
    }.items():
        path = Path(inspect.getsourcefile(function)).resolve()
        dependency_paths[name] = {"path": str(path), "sha256": base.sha256_file(path)}
    summary["provenance"] = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256_file(Path(__file__).resolve()),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "paired_tsv_sha256": base.sha256_file(output_tsv),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_paths,
    }
    try:
        import peptdeep

        summary["provenance"]["peptdeep_version"] = getattr(peptdeep, "__version__", "unknown")
    except Exception as exc:
        summary["provenance"]["peptdeep_version"] = f"unavailable: {exc}"

    base.atomic_json(out_dir / "peakmatch_sensitivity_summary.json", summary)
    checksum_lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{base.sha256_file(path)}  {path.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["matching_diagnostics"], indent=2), flush=True)
    print(json.dumps(summary["scoring_configurations"], indent=2), flush=True)


if __name__ == "__main__":
    main()
