#!/usr/bin/env python3
"""Require exact agreement between the pandas and PyArrow summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = {"D0", "D1", "D2", "D3", "C1", "C2", "C3"}
RUNS = {"P2", "P3"}


def load_tsv(path: Path, key_columns: tuple[str, ...], value_columns: tuple[str, ...]) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for row in rows:
        key = tuple(row[column] for column in key_columns)
        if key in result:
            raise SystemExit(f"duplicate key {key} in {path}")
        result[key] = {column: int(row[column]) for column in value_columns}
    return result


def require_equal(label: str, primary: dict, independent: dict) -> None:
    if primary != independent:
        primary_keys = set(primary)
        independent_keys = set(independent)
        lines = [f"{label} summaries differ"]
        if primary_keys != independent_keys:
            lines.append(f"primary-only keys: {sorted(primary_keys - independent_keys)}")
            lines.append(f"independent-only keys: {sorted(independent_keys - primary_keys)}")
        for key in sorted(primary_keys & independent_keys):
            if primary[key] != independent[key]:
                lines.append(f"{key}: primary={primary[key]} independent={independent[key]}")
        raise SystemExit("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir

    per_values = (
        "report_rows",
        "target_rows",
        "rows_1pct",
        "unique_precursors",
        "unique_peptides",
    )
    primary_per = load_tsv(out / "per_run_counts.tsv", ("arm", "run"), per_values)
    independent_per = load_tsv(
        out / "independent_per_run_counts.tsv", ("arm", "run"), per_values
    )
    require_equal("per-run", primary_per, independent_per)
    if {key[0] for key in primary_per} != ARMS or {key[1] for key in primary_per} != RUNS:
        raise SystemExit("per-run summary does not contain exactly D0-D3/C1-C3 on P2 and P3")

    cross_values = (
        "P2_unique",
        "P3_unique",
        "run_summed",
        "P2_P3_shared",
        "P2_P3_union",
    )
    primary_cross = load_tsv(
        out / "cross_run_counts.tsv", ("arm", "metric"), cross_values
    )
    independent_cross = load_tsv(
        out / "independent_cross_run_counts.tsv", ("arm", "metric"), cross_values
    )
    require_equal("cross-run", primary_cross, independent_cross)

    contrast_values = (
        "base_count",
        "comparison_count",
        "shared",
        "gained",
        "lost",
        "net",
    )
    contrast_key = ("contrast", "base_arm", "comparison_arm", "scope", "metric")
    primary_contrasts = load_tsv(
        out / "paired_contrasts.tsv", contrast_key, contrast_values
    )
    independent_contrasts = load_tsv(
        out / "independent_paired_contrasts.tsv", contrast_key, contrast_values
    )
    require_equal("paired-contrast", primary_contrasts, independent_contrasts)

    for key, values in primary_contrasts.items():
        if values["comparison_count"] - values["base_count"] != values["net"]:
            raise SystemExit(f"net-count identity failed for {key}")
        if values["gained"] - values["lost"] != values["net"]:
            raise SystemExit(f"gained/lost identity failed for {key}")
        if values["shared"] + values["gained"] != values["comparison_count"]:
            raise SystemExit(f"comparison-set identity failed for {key}")
        if values["shared"] + values["lost"] != values["base_count"]:
            raise SystemExit(f"base-set identity failed for {key}")

    result = {
        "status": "PASS",
        "implementations": ["pandas", "pyarrow with Python sets"],
        "per_run_cells_verified": len(primary_per),
        "cross_run_cells_verified": len(primary_cross),
        "contrast_cells_verified": len(primary_contrasts),
        "expected_arms": sorted(ARMS),
        "expected_evaluation_runs": sorted(RUNS),
        "p1_absent_from_primary_evaluation": True,
    }
    path = out / "SUMMARY_VERIFICATION.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite: {path}")
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
