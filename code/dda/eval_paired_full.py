#!/usr/bin/env python3
"""Paired full-corpus APD evaluation for the JASMS manuscript.

Run this script from ``/path/to/peptdeep_project`` after copying it into that project's
``scripts`` directory. It deliberately reuses the project's existing MGF parser,
ground-truth construction, model loader, prediction function, and metric
definitions. Unlike the historical evaluator, it:

* requires an explicit dataset label;
* evaluates stock and fine-tuned models on exactly the same spectrum IDs;
* exports one paired row per selected spectrum;
* distinguishes MGF entries, parser records, eligible rows, selected rows, and
  successfully scored pairs;
* calculates stratified paired-bootstrap confidence intervals; and
* records input and checkpoint SHA-256 checksums.

It never overwrites an existing output directory unless ``--allow_existing`` is
given. Even with that flag, archival result files are written atomically.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_all import angular_similarity, convert_mod, parse_mgf_file, pearson_corr
from eval_koina_bacterial import compute_fragment_mz, compute_gt_z1
from eval_local_zeroshot_baseline import (
    _load_finetuned_ms2,
    cosine,
    predict_species,
    spearman_obs_topN,
)
from rescore_spearman_union import spearman_union_topN


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
METRICS = (
    "spearman_union_top7",
    "spearman_obs_top7",
    "spectral_angle",
    "cosine",
    "pcc",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Paired stock-versus-fine-tuned APD evaluation with per-spectrum output"
    )
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_label", required=True)
    ap.add_argument("--finetune_ckpt", required=True)
    ap.add_argument("--checkpoint_label", required=True)
    ap.add_argument("--max_spectra", type=int, required=True,
                    help="Per-file parser cap; use 0 for no cap")
    ap.add_argument("--test_ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topn", type=int, default=7)
    ap.add_argument("--prediction_batch_size", type=int, default=2000)
    ap.add_argument("--stock_nce", type=float, default=30.0,
                    help="Prediction NCE for the installed stock model")
    ap.add_argument("--finetune_nce", type=float, default=30.0,
                    help="Prediction NCE for the fine-tuned model")
    ap.add_argument("--bootstrap_reps", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260816)
    ap.add_argument("--bootstrap_block", type=int, default=32)
    ap.add_argument("--allow_existing", action="store_true")
    args = ap.parse_args()

    if not 0 < args.test_ratio <= 1:
        ap.error("--test_ratio must be in (0, 1]")
    if args.max_spectra < 0:
        ap.error("--max_spectra must be >= 0")
    if args.topn != 7:
        ap.error("The manuscript metric is fixed at topn=7")
    if args.prediction_batch_size < 1:
        ap.error("--prediction_batch_size must be positive")
    if not 0 < args.stock_nce <= 100 or not 0 < args.finetune_nce <= 100:
        ap.error("prediction NCE values must be in (0, 100]")
    if args.bootstrap_reps < 1 or args.bootstrap_block < 1:
        ap.error("bootstrap counts must be positive")
    return args


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def count_mgf_entries(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip() == "BEGIN IONS":
                count += 1
    return count


def load_file_with_ids(mgf: Path, max_spectra: int) -> tuple[list[dict], dict]:
    """Mirror ``load_species_file`` while retaining parser scan IDs and counts."""
    mgf_entries = count_mgf_entries(mgf)
    spectra = parse_mgf_file(str(mgf), max_spectra or None)
    excluded = Counter()
    rows: list[dict] = []

    for spectrum in spectra:
        seq, mods, sites = convert_mod(spectrum.get("SEQ", ""))
        if not seq:
            excluded["empty_sequence"] += 1
            continue
        if not all(aa in VALID_AA for aa in seq):
            excluded["invalid_amino_acid"] += 1
            continue
        if len(seq) < 7 or len(seq) > 30:
            excluded["peptide_length_outside_7_30"] += 1
            continue
        if "Unknown" in mods:
            excluded["unsupported_modification"] += 1
            continue
        raw_charge = spectrum.get("CHARGE", "2+").replace("+", "").replace("-", "")
        try:
            charge = int(raw_charge)
        except ValueError:
            excluded["invalid_charge"] += 1
            continue
        if charge < 1 or charge > 6:
            excluded["charge_outside_1_6"] += 1
            continue

        scan = int(spectrum["_scan"])
        source = str(spectrum["_file"])
        rows.append(
            {
                "spectrum_id": f"{source}:{scan}",
                "source": source,
                "scan": scan,
                "sequence": seq,
                "mods": mods,
                "mod_sites": sites,
                "charge": charge,
                "nce": 30.0,
                "instrument": "QE",
                "exp_mz": spectrum["mz"],
                "exp_intensity": spectrum["intensity"],
            }
        )

    counts = {
        "file": str(mgf.resolve()),
        "sha256": sha256_file(mgf),
        "size_bytes": mgf.stat().st_size,
        "mgf_entries": mgf_entries,
        "parser_records": len(spectra),
        "eligible_rows": len(rows),
        "not_returned_by_parser_or_beyond_cap": mgf_entries - len(spectra),
        "filter_exclusions": dict(sorted(excluded.items())),
    }
    return rows, counts


def select_rows(
    mgf_files: list[Path], max_spectra: int, test_ratio: float, seed: int
) -> tuple[dict[str, list[dict]], list[dict]]:
    selected: dict[str, list[dict]] = {}
    manifests: list[dict] = []

    for file_index, mgf in enumerate(mgf_files):
        rows, counts = load_file_with_ids(mgf, max_spectra)
        if not rows:
            raise RuntimeError(f"No eligible rows in {mgf}")

        # Preserve the historical seed+file_index selection exactly. At ratio 1.0
        # this permutes all rows but does not discard any.
        np.random.seed(seed + file_index)
        permutation = np.random.permutation(len(rows))
        n_selected = max(1, int(len(rows) * test_ratio))
        chosen = [rows[int(i)] for i in permutation[:n_selected]]
        for rank, row in enumerate(chosen):
            row["selection_rank"] = rank

        counts["selected_rows"] = len(chosen)
        counts["selection_seed"] = seed + file_index
        manifests.append(counts)
        selected[mgf.stem] = chosen
        print(
            f"[{mgf.stem}] entries={counts['mgf_entries']} "
            f"parsed={counts['parser_records']} eligible={len(rows)} "
            f"selected={len(chosen)}",
            flush=True,
        )
    return selected, manifests


def empty_metric(status: str) -> dict:
    return {
        "status": status,
        "spearman_union_top7": np.nan,
        "spearman_obs_top7": np.nan,
        "spectral_angle": np.nan,
        "cosine": np.nan,
        "pcc": np.nan,
        "pcc90": np.nan,
        "union_size": np.nan,
    }


def score_prediction(row: dict, precursor_row: pd.Series, fragment_df: pd.DataFrame, topn: int) -> dict:
    start = int(precursor_row["frag_start_idx"])
    stop = int(precursor_row["frag_stop_idx"])
    n_ions = stop - start
    if n_ions <= 0:
        return empty_metric("no_predicted_fragment_rows")

    b_mz, y_mz = compute_fragment_mz(row["sequence"], row["mods"], row["mod_sites"])
    gt = compute_gt_z1(row, b_mz[:n_ions], y_mz[:n_ions])
    if len(gt) != 2 * n_ions:
        return empty_metric("ground_truth_length_mismatch")
    if len(gt) == 0 or np.max(gt) <= 0:
        return empty_metric("zero_ground_truth")

    pred_b = fragment_df.iloc[start:stop]["b_z1"].values[:n_ions].astype(float)
    pred_y = fragment_df.iloc[start:stop]["y_z1"].values[:n_ions].astype(float)
    pred = np.concatenate([pred_b, pred_y[::-1]])
    if len(pred) != len(gt):
        return empty_metric("prediction_length_mismatch")
    if len(pred) == 0 or np.max(pred) <= 0:
        return empty_metric("zero_prediction")
    pred = pred / np.max(pred)

    pcc = pearson_corr(pred, gt)
    union_spearman, union_size = spearman_union_topN(gt, pred, topn)
    return {
        "status": "ok",
        "spearman_union_top7": float(union_spearman),
        "spearman_obs_top7": float(spearman_obs_topN(gt, pred, topn)),
        "spectral_angle": float(angular_similarity(pred, gt)),
        "cosine": float(cosine(pred, gt)),
        "pcc": float(pcc) if pcc == pcc else np.nan,
        "pcc90": float(pcc >= 0.9) if pcc == pcc else np.nan,
        "union_size": int(union_size),
    }


def evaluate_model(
    model_manager,
    selected: dict[str, list[dict]],
    topn: int,
    prediction_batch_size: int,
    label: str,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for source, rows in selected.items():
        print(f"[{label}] {source}: {len(rows)} selected spectra", flush=True)
        for offset in range(0, len(rows), prediction_batch_size):
            batch = rows[offset : offset + prediction_batch_size]
            precursor_df, fragment_df = predict_species(model_manager, batch)
            if "b_z1" not in fragment_df.columns:
                raise RuntimeError(f"b_z1 missing from prediction columns: {list(fragment_df.columns)}")
            if len(precursor_df) != len(batch):
                raise RuntimeError(
                    f"Prediction row mismatch for {source}: {len(precursor_df)} != {len(batch)}"
                )
            for index, row in enumerate(batch):
                results[row["spectrum_id"]] = score_prediction(
                    row, precursor_df.iloc[index], fragment_df, topn
                )
            del precursor_df, fragment_df, batch
            gc.collect()
            print(
                f"[{label}] {source}: {min(offset + prediction_batch_size, len(rows))}/"
                f"{len(rows)}",
                flush=True,
            )
    return results


def release_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_paired_frame(
    selected: dict[str, list[dict]], stock: dict[str, dict], fine: dict[str, dict],
    stock_nce: float = 30.0, finetune_nce: float = 30.0,
) -> pd.DataFrame:
    records: list[dict] = []
    for source, rows in selected.items():
        for row in rows:
            spectrum_id = row["spectrum_id"]
            if spectrum_id not in stock or spectrum_id not in fine:
                raise RuntimeError(f"Missing paired prediction for {spectrum_id}")
            rec = {
                "spectrum_id": spectrum_id,
                "source": source,
                "scan": row["scan"],
                "selection_rank": row["selection_rank"],
                "sequence": row["sequence"],
                "mods": row["mods"],
                "mod_sites": row["mod_sites"],
                "charge": row["charge"],
                "instrument": row["instrument"],
                "acquisition_nce": row["nce"],
                "stock_prediction_nce": stock_nce,
                "finetune_prediction_nce": finetune_nce,
            }
            rec.update({f"stock_{key}": value for key, value in stock[spectrum_id].items()})
            rec.update({f"fine_{key}": value for key, value in fine[spectrum_id].items()})
            rec["valid_pair"] = (
                rec["stock_status"] == "ok" and rec["fine_status"] == "ok"
            )
            for metric in METRICS:
                rec[f"delta_{metric}"] = rec[f"fine_{metric}"] - rec[f"stock_{metric}"]
            rec["delta_pcc90"] = rec["fine_pcc90"] - rec["stock_pcc90"]
            records.append(rec)
    frame = pd.DataFrame.from_records(records)
    if frame["spectrum_id"].duplicated().any():
        duplicates = frame.loc[frame["spectrum_id"].duplicated(), "spectrum_id"].tolist()[:5]
        raise RuntimeError(f"Duplicate spectrum IDs: {duplicates}")
    return frame


def finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def summarize_model(frame: pd.DataFrame, prefix: str) -> dict:
    valid = frame.loc[frame["valid_pair"]].copy()
    return {
        "n_valid_pair": int(len(valid)),
        "n_spearman_union": int(valid[f"{prefix}_spearman_union_top7"].notna().sum()),
        "spearman_union_top7_mean": finite_mean(valid[f"{prefix}_spearman_union_top7"]),
        "spearman_obs_top7_mean": finite_mean(valid[f"{prefix}_spearman_obs_top7"]),
        "spectral_angle_mean": finite_mean(valid[f"{prefix}_spectral_angle"]),
        "cosine_mean": finite_mean(valid[f"{prefix}_cosine"]),
        "pcc_mean": finite_mean(valid[f"{prefix}_pcc"]),
        "pcc90": finite_mean(valid[f"{prefix}_pcc90"]),
        "union_size_mean": finite_mean(valid[f"{prefix}_union_size"]),
    }


def stratified_paired_bootstrap(
    frame: pd.DataFrame,
    delta_column: str,
    reps: int,
    seed: int,
    block_size: int,
    scale: float = 1.0,
) -> dict:
    valid = frame.loc[frame["valid_pair"], ["source", delta_column]].copy()
    valid[delta_column] = pd.to_numeric(valid[delta_column], errors="coerce")
    valid = valid[np.isfinite(valid[delta_column])]
    groups = [
        group[delta_column].to_numpy(dtype=float)
        for _, group in valid.groupby("source", sort=True)
        if len(group)
    ]
    if not groups:
        raise RuntimeError(f"No finite paired values for {delta_column}")

    total_n = sum(len(group) for group in groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for start in range(0, reps, block_size):
        current = min(block_size, reps - start)
        sums = np.zeros(current, dtype=float)
        for group in groups:
            indices = rng.integers(0, len(group), size=(current, len(group)))
            sums += group[indices].sum(axis=1)
        estimates[start : start + current] = sums / total_n

    point = float(valid[delta_column].mean()) * scale
    low, high = np.quantile(estimates * scale, [0.025, 0.975])
    return {
        "estimand": "fine_tuned_minus_stock, spectrum-weighted mean",
        "stratification": "source MGF; fixed observed count per source",
        "interpretation": (
            "spectrum-level paired uncertainty; not biological-replicate uncertainty"
        ),
        "n": int(total_n),
        "bootstrap_replicates": int(reps),
        "point_estimate": point,
        "ci95_percentile": [float(low), float(high)],
    }


def status_counts(frame: pd.DataFrame, prefix: str) -> dict:
    return {
        str(key): int(value)
        for key, value in frame[f"{prefix}_status"].value_counts(dropna=False).items()
    }


def make_summary(frame: pd.DataFrame, args: argparse.Namespace, manifests: list[dict]) -> dict:
    valid = frame.loc[frame["valid_pair"]]
    per_source = {}
    for source, group in frame.groupby("source", sort=True):
        per_source[source] = {
            "n_selected": int(len(group)),
            "n_valid_pair": int(group["valid_pair"].sum()),
            "stock": summarize_model(group, "stock"),
            "fine_tuned": summarize_model(group, "fine"),
            "difference": {
                "spearman_union_top7_mean": finite_mean(
                    group.loc[group["valid_pair"], "delta_spearman_union_top7"]
                ),
                "pcc90_percentage_points": (
                    finite_mean(group.loc[group["valid_pair"], "delta_pcc90"]) * 100.0
                    if len(group.loc[group["valid_pair"]])
                    else None
                ),
            },
        }

    spearman_ci = stratified_paired_bootstrap(
        frame,
        "delta_spearman_union_top7",
        args.bootstrap_reps,
        args.bootstrap_seed,
        args.bootstrap_block,
    )
    pcc90_ci = stratified_paired_bootstrap(
        frame,
        "delta_pcc90",
        args.bootstrap_reps,
        args.bootstrap_seed + 1,
        args.bootstrap_block,
        scale=100.0,
    )
    pcc90_ci["units"] = "percentage points"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset_label,
        "comparison": f"AlphaPeptDeep installed v3 stock vs {args.checkpoint_label}",
        "headline_metric": "mean Spearman over union of observed and predicted top-7 ions",
        "config": {
            "data_dir": str(Path(os.path.expanduser(args.data_dir)).resolve()),
            "finetune_checkpoint": str(Path(os.path.expanduser(args.finetune_ckpt)).resolve()),
            "checkpoint_label": args.checkpoint_label,
            "instrument": "QE",
            "acquisition_nce": 30.0,
            "stock_prediction_nce": args.stock_nce,
            "finetune_prediction_nce": args.finetune_nce,
            "fragment_tolerance_Da": 0.5,
            "normalization": "divide each observed/predicted vector by its maximum",
            "peptide_length": "7-30",
            "charge": "1-6",
            "max_spectra_per_file": args.max_spectra,
            "test_ratio": args.test_ratio,
            "selection_seed": args.seed,
            "prediction_batch_size": args.prediction_batch_size,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_reps,
            "overall_weighting": "spectrum-weighted; bootstrap stratified by source MGF",
        },
        "denominators": {
            "mgf_entries": int(sum(item["mgf_entries"] for item in manifests)),
            "parser_records": int(sum(item["parser_records"] for item in manifests)),
            "eligible_rows": int(sum(item["eligible_rows"] for item in manifests)),
            "selected_rows": int(len(frame)),
            "stock_status": status_counts(frame, "stock"),
            "fine_status": status_counts(frame, "fine"),
            "valid_pairs": int(frame["valid_pair"].sum()),
        },
        "overall": {
            "stock": summarize_model(valid, "stock"),
            "fine_tuned": summarize_model(valid, "fine"),
            "difference": {
                "spearman_union_top7": spearman_ci,
                "pcc90": pcc90_ci,
            },
        },
        "per_source": per_source,
        "input_files": manifests,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    checkpoint = Path(os.path.expanduser(args.finetune_ckpt)).resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: data directory not found: {data_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}")
    mgf_files = sorted(data_dir.glob("*.mgf"))
    if not mgf_files:
        raise SystemExit(f"ERROR: no MGF files in {data_dir}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_existing:
        raise SystemExit(
            f"ERROR: non-empty output directory exists: {out_dir}; choose a new path"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_label}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Checkpoint SHA-256: {sha256_file(checkpoint)}", flush=True)
    from peptdeep.pretrained_models import ModelManager

    print("Loading stock AlphaPeptDeep model", flush=True)
    stock_manager = ModelManager()
    stock_manager.load_installed_models()
    print(f"Loading fine-tuned model: {args.checkpoint_label}", flush=True)
    fine_manager = ModelManager()
    fine_manager.load_installed_models()
    _load_finetuned_ms2(fine_manager, str(checkpoint))

    # Process one MGF at a time. Peak arrays dominate CPU memory, especially for
    # the 73,529-spectrum external dataset; retaining all files at once is both
    # unnecessary and risky. Metric-only shard frames remain small.
    shard_dir = out_dir / "paired_shards"
    shard_dir.mkdir(exist_ok=False)
    manifests: list[dict] = []
    paired_frames: list[pd.DataFrame] = []
    for file_index, mgf in enumerate(mgf_files):
        rows, counts = load_file_with_ids(mgf, args.max_spectra)
        if not rows:
            raise RuntimeError(f"No eligible rows in {mgf}")
        np.random.seed(args.seed + file_index)
        permutation = np.random.permutation(len(rows))
        n_selected = max(1, int(len(rows) * args.test_ratio))
        chosen = [rows[int(i)] for i in permutation[:n_selected]]
        for rank, row in enumerate(chosen):
            row["selection_rank"] = rank
        counts["selected_rows"] = len(chosen)
        counts["selection_seed"] = args.seed + file_index
        manifests.append(counts)
        selected_file = {mgf.stem: chosen}
        selected_stock = {
            mgf.stem: [{**row, "nce": args.stock_nce} for row in chosen]
        }
        selected_fine = {
            mgf.stem: [{**row, "nce": args.finetune_nce} for row in chosen]
        }
        print(
            f"[{mgf.stem}] entries={counts['mgf_entries']} "
            f"parsed={counts['parser_records']} eligible={len(rows)} "
            f"selected={len(chosen)}",
            flush=True,
        )

        stock_results = evaluate_model(
            stock_manager,
            selected_stock,
            args.topn,
            args.prediction_batch_size,
            "stock",
        )
        fine_results = evaluate_model(
            fine_manager,
            selected_fine,
            args.topn,
            args.prediction_batch_size,
            "fine",
        )
        shard_frame = build_paired_frame(
            selected_file, stock_results, fine_results,
            args.stock_nce, args.finetune_nce,
        )
        shard_path = shard_dir / f"{file_index:02d}_{mgf.stem}.tsv"
        shard_tmp = shard_path.with_suffix(".tsv.tmp")
        shard_frame.to_csv(shard_tmp, sep="\t", index=False, na_rep="NA")
        shard_tmp.replace(shard_path)
        paired_frames.append(shard_frame)

        del rows, chosen, selected_file, selected_stock, selected_fine
        del stock_results, fine_results, shard_frame
        gc.collect()

    del stock_manager
    del fine_manager
    release_gpu_cache()

    frame = pd.concat(paired_frames, ignore_index=True)
    del paired_frames
    tsv_path = out_dir / "paired_per_spectrum.tsv"
    tsv_tmp = out_dir / "paired_per_spectrum.tsv.tmp"
    frame.to_csv(tsv_tmp, sep="\t", index=False, na_rep="NA")
    tsv_tmp.replace(tsv_path)

    summary = make_summary(frame, args, manifests)
    summary["provenance"] = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "python": sys.version,
        "platform": platform.platform(),
        "paired_tsv_sha256": sha256_file(tsv_path),
    }
    try:
        import peptdeep

        summary["provenance"]["peptdeep_version"] = getattr(peptdeep, "__version__", "unknown")
    except Exception as exc:
        summary["provenance"]["peptdeep_version"] = f"unavailable: {exc}"

    summary_path = out_dir / "paired_summary.json"
    atomic_json(summary_path, summary)

    checksum_lines = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "SHA256SUMS.txt.tmp"}:
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    checksum_tmp = out_dir / "SHA256SUMS.txt.tmp"
    checksum_tmp.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    checksum_tmp.replace(out_dir / "SHA256SUMS.txt")

    overall = summary["overall"]
    print("\nFINAL PAIRED SUMMARY", flush=True)
    print(json.dumps(overall, indent=2), flush=True)
    print(f"Wrote {tsv_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {out_dir / 'SHA256SUMS.txt'}", flush=True)


if __name__ == "__main__":
    main()
