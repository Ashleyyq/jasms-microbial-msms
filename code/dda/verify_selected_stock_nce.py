#!/usr/bin/env python3
"""Reproduce the frozen stock-NCE validation point in a fresh model process."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import eval_paired_full as paired
import eval_stock_nce_sweep as sweep


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--val_fold_list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prediction_batch_size", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_path = Path(args.selection).resolve()
    val_fold_list = Path(args.val_fold_list).resolve()
    out = Path(args.out).resolve()
    for path in (selection_path, val_fold_list):
        if not path.is_file():
            raise SystemExit(f"ERROR: missing input: {path}")
    if out.exists():
        raise SystemExit(f"ERROR: output exists: {out}")

    frozen = json.loads(selection_path.read_text())
    if frozen.get("selection_status") != "frozen_validation_only":
        raise RuntimeError("Selection is not frozen validation-only")
    if frozen.get("boundary_selected") is not False:
        raise RuntimeError("Selection is a grid-boundary result")
    if frozen.get("n_validation_spectra") != 49_000:
        raise RuntimeError("Unexpected validation request denominator")
    if frozen.get("selected_n_valid") != 48_999:
        raise RuntimeError("Unexpected selected-candidate valid-score denominator")
    if frozen.get("fresh_stock_model_per_candidate") is not True:
        raise RuntimeError("Frozen selection did not use a fresh stock model per candidate")
    selected_nce = float(frozen["selected_stock_prediction_nce"])
    selected_key = f"{selected_nce:g}"
    expected_summary = frozen["results"][selected_key]

    loader_args = SimpleNamespace(
        val_fold_list=str(val_fold_list),
        exclude_source_prefix=frozen["excluded_source_prefix"],
        max_per_file=int(frozen["max_per_file"]),
        seed=int(frozen["seed"]),
    )
    selected, manifest = sweep.load_validation(loader_args)
    n_requests = sum(item["selected_rows"] for item in manifest)
    if n_requests != 49_000:
        raise RuntimeError(f"Fresh-process request denominator is {n_requests}")
    for rows in selected.values():
        for row in rows:
            row["nce"] = selected_nce

    from peptdeep.pretrained_models import ModelManager

    manager = ModelManager()
    manager.load_installed_models()
    predictions = paired.evaluate_model(
        manager,
        selected,
        topn=7,
        prediction_batch_size=args.prediction_batch_size,
        label=f"fresh_process_stock_nce_{selected_key}",
    )
    observed_summary = sweep.summarize(predictions)
    valid_sum = sum(
        float(record["spearman_union_top7"])
        for record in predictions.values()
        if record["status"] == "ok"
    )
    observed_summary["spearman_union_top7_mean_fixed_denominator"] = valid_sum / 49_000
    observed_summary["fixed_denominator_non_ok_scored_zero"] = (
        49_000 - observed_summary["n_valid"]
    )

    differences: dict[str, float | bool] = {}
    exact = True
    for key, expected in expected_summary.items():
        observed = observed_summary.get(key)
        if isinstance(expected, dict):
            same = observed == expected
            differences[key] = same
            exact = exact and same
        elif expected is None:
            same = observed is None
            differences[key] = same
            exact = exact and same
        else:
            delta = abs(float(observed) - float(expected))
            differences[key] = delta
            exact = exact and delta <= 1e-12

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "exact_reproduction" if exact else "mismatch",
        "fresh_model_process": True,
        "selected_stock_prediction_nce": selected_nce,
        "expected_summary": expected_summary,
        "observed_summary": observed_summary,
        "absolute_differences_or_exact_flags": differences,
        "n_validation_requests": n_requests,
        "inputs": {
            "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "val_fold_list": {"path": str(val_fold_list), "sha256": sha256_file(val_fold_list)},
        },
        "scripts": {
            "verification": sha256_file(Path(__file__).resolve()),
            "sweep_module": sha256_file(Path(sweep.__file__).resolve()),
            "paired_evaluator": sha256_file(Path(paired.__file__).resolve()),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(out)
    print(json.dumps(payload, indent=2), flush=True)
    if not exact:
        raise SystemExit("ERROR: fresh-process selected-NCE result did not reproduce")


if __name__ == "__main__":
    main()
