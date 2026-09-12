#!/usr/bin/env python3
"""Independent PyArrow/set implementation of the final P2/P3 DIA counts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq


ARMS = ("D0", "D1", "D2", "D3", "C1", "C2", "C3")
RUNS = ("P2", "P3")
CONTRASTS = (("D3_minus_D2", "D2", "D3"), ("C3_minus_C2", "C2", "C3"))


def report_path(root: Path, arm: str, run: str) -> Path:
    layer = "direct" if arm[0] == "D" else "carafe_search"
    return root / layer / arm / f"test_{run}" / "report.parquet"


def is_decoy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "decoy"}


def read_sets(path: Path) -> tuple[dict, dict[str, set[str]]]:
    if not path.is_file():
        raise SystemExit(f"missing report: {path}")
    schema_names = set(pq.read_schema(path).names)
    required = {"Q.Value", "Global.Q.Value", "Precursor.Id", "Stripped.Sequence"}
    missing = sorted(required - schema_names)
    if missing:
        raise SystemExit(f"missing required columns in {path}: {missing}")
    columns = ["Q.Value", "Global.Q.Value", "Precursor.Id", "Stripped.Sequence"]
    if "Decoy" in schema_names:
        columns.append("Decoy")
    data = pq.read_table(path, columns=columns).to_pydict()

    precursor_set: set[str] = set()
    peptide_set: set[str] = set()
    report_rows = len(data["Q.Value"])
    target_rows = 0
    kept_rows = 0
    decoys = data.get("Decoy", [False] * report_rows)
    for q_value, global_q, precursor, peptide, decoy in zip(
        data["Q.Value"],
        data["Global.Q.Value"],
        data["Precursor.Id"],
        data["Stripped.Sequence"],
        decoys,
    ):
        if is_decoy(decoy):
            continue
        target_rows += 1
        if q_value is None or global_q is None:
            continue
        if float(q_value) > 0.01 or float(global_q) > 0.01:
            continue
        kept_rows += 1
        if precursor is not None:
            precursor_set.add(str(precursor))
        if peptide is not None:
            peptide_set.add(str(peptide))
    stats = {
        "report_rows": report_rows,
        "target_rows": target_rows,
        "rows_1pct": kept_rows,
        "unique_precursors": len(precursor_set),
        "unique_peptides": len(peptide_set),
    }
    return stats, {"precursors": precursor_set, "peptides": peptide_set}


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite: {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        raise SystemExit(f"primary output directory does not exist: {args.output_dir}")

    sets: dict[str, dict[str, dict[str, set[str]]]] = {}
    per_run_rows: list[dict] = []
    for arm in ARMS:
        sets[arm] = {}
        for run in RUNS:
            stats, report_sets = read_sets(report_path(args.run_root, arm, run))
            sets[arm][run] = report_sets
            per_run_rows.append({"arm": arm, "run": run, **stats})
    per_run_rows.sort(key=lambda row: (row["arm"], row["run"]))
    write_rows(
        args.output_dir / "independent_per_run_counts.tsv",
        [
            "arm",
            "run",
            "report_rows",
            "target_rows",
            "rows_1pct",
            "unique_precursors",
            "unique_peptides",
        ],
        per_run_rows,
    )

    union_sets: dict[str, dict[str, set[str]]] = {}
    cross_rows: list[dict] = []
    for arm in ARMS:
        union_sets[arm] = {}
        for metric in ("peptides", "precursors"):
            p2 = sets[arm]["P2"][metric]
            p3 = sets[arm]["P3"][metric]
            union_sets[arm][metric] = p2 | p3
            cross_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "P2_unique": len(p2),
                    "P3_unique": len(p3),
                    "run_summed": len(p2) + len(p3),
                    "P2_P3_shared": len(p2 & p3),
                    "P2_P3_union": len(p2 | p3),
                }
            )
    cross_rows.sort(key=lambda row: (row["arm"], row["metric"]))
    write_rows(
        args.output_dir / "independent_cross_run_counts.tsv",
        [
            "arm",
            "metric",
            "P2_unique",
            "P3_unique",
            "run_summed",
            "P2_P3_shared",
            "P2_P3_union",
        ],
        cross_rows,
    )

    contrast_rows: list[dict] = []
    for label, base_arm, comparison_arm in CONTRASTS:
        for metric in ("peptides", "precursors"):
            for scope in (*RUNS, "union"):
                if scope == "union":
                    base = union_sets[base_arm][metric]
                    comparison = union_sets[comparison_arm][metric]
                else:
                    base = sets[base_arm][scope][metric]
                    comparison = sets[comparison_arm][scope][metric]
                contrast_rows.append(
                    {
                        "contrast": label,
                        "base_arm": base_arm,
                        "comparison_arm": comparison_arm,
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
    contrast_rows.sort(key=lambda row: (row["contrast"], row["metric"], row["scope"]))
    write_rows(
        args.output_dir / "independent_paired_contrasts.tsv",
        [
            "contrast",
            "base_arm",
            "comparison_arm",
            "scope",
            "metric",
            "base_count",
            "comparison_count",
            "shared",
            "gained",
            "lost",
            "net",
        ],
        contrast_rows,
    )


if __name__ == "__main__":
    main()
