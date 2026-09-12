#!/usr/bin/env python3
"""Summarize one Carafe seed with independent pandas and PyArrow readers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ARMS = ("C1", "C2", "C3")
RUNS = ("P2", "P3")
METRICS = ("peptides", "precursors")


def report_path(root: Path, arm: str, run: str) -> Path:
    return root / "carafe_search" / arm / f"test_{run}" / "report.parquet"


def clean_identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if not text or text.lower() == "nan" else text


def pandas_decoy_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).ne(0)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "decoy"})


def pandas_sets(path: Path) -> dict[str, set[str]]:
    frame = pd.read_parquet(path)
    required = {"Q.Value", "Global.Q.Value", "Precursor.Id", "Stripped.Sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    if "Decoy" in frame.columns:
        decoy = pandas_decoy_mask(frame["Decoy"])
        frame = frame.loc[~decoy]
    frame = frame.loc[
        frame["Q.Value"].le(0.01) & frame["Global.Q.Value"].le(0.01)
    ]
    return {
        "peptides": {
            value
            for item in frame["Stripped.Sequence"]
            if (value := clean_identifier(item)) is not None
        },
        "precursors": {
            value
            for item in frame["Precursor.Id"]
            if (value := clean_identifier(item)) is not None
        },
    }


def is_decoy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "decoy"}


def pyarrow_sets(path: Path) -> dict[str, set[str]]:
    schema = set(pq.read_schema(path).names)
    columns = ["Q.Value", "Global.Q.Value", "Precursor.Id", "Stripped.Sequence"]
    if "Decoy" in schema:
        columns.append("Decoy")
    data = pq.read_table(path, columns=columns).to_pydict()
    decoys = data.get("Decoy", [False] * len(data["Q.Value"]))
    result = {"peptides": set(), "precursors": set()}
    for q_value, global_q, precursor, peptide, decoy in zip(
        data["Q.Value"],
        data["Global.Q.Value"],
        data["Precursor.Id"],
        data["Stripped.Sequence"],
        decoys,
    ):
        if is_decoy(decoy) or q_value is None or global_q is None:
            continue
        if float(q_value) > 0.01 or float(global_q) > 0.01:
            continue
        if (value := clean_identifier(peptide)) is not None:
            result["peptides"].add(value)
        if (value := clean_identifier(precursor)) is not None:
            result["precursors"].add(value)
    return result


def count_rows(all_sets: dict[str, dict[str, dict[str, set[str]]]]) -> tuple[list[dict], list[dict]]:
    per_run: list[dict] = []
    cross_run: list[dict] = []
    for arm in ARMS:
        for run in RUNS:
            per_run.append(
                {
                    "arm": arm,
                    "run": run,
                    "unique_peptides": len(all_sets[arm][run]["peptides"]),
                    "unique_precursors": len(all_sets[arm][run]["precursors"]),
                }
            )
        for metric in METRICS:
            p2 = all_sets[arm]["P2"][metric]
            p3 = all_sets[arm]["P3"][metric]
            cross_run.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "P2_unique": len(p2),
                    "P3_unique": len(p3),
                    "P2_P3_shared": len(p2 & p3),
                    "P2_P3_union": len(p2 | p3),
                }
            )
    return per_run, cross_run


def write_tsv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite: {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-root", required=True, type=Path)
    parser.add_argument("--carafe-seed", required=True, type=int)
    args = parser.parse_args()
    output = args.sensitivity_root / "summary"
    output.mkdir(parents=True, exist_ok=False)

    pandas_all: dict[str, dict[str, dict[str, set[str]]]] = {}
    arrow_all: dict[str, dict[str, dict[str, set[str]]]] = {}
    for arm in ARMS:
        pandas_all[arm] = {}
        arrow_all[arm] = {}
        for run in RUNS:
            path = report_path(args.sensitivity_root, arm, run)
            pandas_all[arm][run] = pandas_sets(path)
            arrow_all[arm][run] = pyarrow_sets(path)
            if pandas_all[arm][run] != arrow_all[arm][run]:
                raise SystemExit(f"pandas/PyArrow set mismatch for {arm} {run}")

    per_run, cross_run = count_rows(pandas_all)
    write_tsv(output / "per_run_counts.tsv", per_run)
    write_tsv(output / "cross_run_counts.tsv", cross_run)

    contrasts: list[dict] = []
    for metric in METRICS:
        for scope in (*RUNS, "union"):
            if scope == "union":
                base = pandas_all["C2"]["P2"][metric] | pandas_all["C2"]["P3"][metric]
                comparison = (
                    pandas_all["C3"]["P2"][metric]
                    | pandas_all["C3"]["P3"][metric]
                )
            else:
                base = pandas_all["C2"][scope][metric]
                comparison = pandas_all["C3"][scope][metric]
            contrasts.append(
                {
                    "contrast": "C3_minus_C2",
                    "scope": scope,
                    "metric": metric,
                    "base_count": len(base),
                    "comparison_count": len(comparison),
                    "shared": len(base & comparison),
                    "gained": len(comparison - base),
                    "lost": len(base - comparison),
                    "net": len(comparison) - len(base),
                }
            )
    write_tsv(output / "paired_contrasts.tsv", contrasts)
    verification = {
        "status": "PASS",
        "carafe_seed": args.carafe_seed,
        "implementations": ["pandas", "PyArrow with Python sets"],
        "arms": list(ARMS),
        "training_run": "P1",
        "evaluation_runs": list(RUNS),
        "q_value_filter": "Q.Value <= 0.01 and Global.Q.Value <= 0.01",
        "per_run_cells_verified": len(per_run),
        "cross_run_cells_verified": len(cross_run),
        "contrast_cells_verified": len(contrasts),
    }
    (output / "SUMMARY_VERIFICATION.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
