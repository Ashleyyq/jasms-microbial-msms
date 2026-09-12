#!/usr/bin/env python3
"""Combine the primary and two prespecified Carafe seed summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_SEEDS = (2024, 2025, 2026)
ARMS = ("C1", "C2", "C3")
METRICS = ("peptides", "precursors")
SCOPES = ("P2", "P3", "union")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verification(summary_dir: Path) -> dict:
    path = summary_dir / "SUMMARY_VERIFICATION.json"
    result = json.loads(path.read_text())
    if result.get("status") != "PASS":
        raise SystemExit(f"summary verification did not pass: {path}")
    return result


def load_cross_run(summary_dir: Path) -> dict[tuple[str, str], dict[str, int]]:
    path = summary_dir / "cross_run_counts.tsv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        key = (row["arm"], row["metric"])
        result[key] = {
            "P2": int(row["P2_unique"]),
            "P3": int(row["P3_unique"]),
            "union": int(row["P2_P3_union"]),
        }
    return result


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sign(value: int) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-summary", required=True, type=Path)
    parser.add_argument(
        "--seed-summary",
        action="append",
        required=True,
        help="SEED=PATH, repeated for seeds 2025 and 2026",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    seed_dirs = {2024: args.primary_summary}
    for item in args.seed_summary:
        seed_text, separator, path_text = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid --seed-summary: {item}")
        seed = int(seed_text)
        if seed in seed_dirs:
            raise SystemExit(f"duplicate seed: {seed}")
        seed_dirs[seed] = Path(path_text)
    if tuple(sorted(seed_dirs)) != EXPECTED_SEEDS:
        raise SystemExit(
            f"expected exactly seeds {EXPECTED_SEEDS}, observed {sorted(seed_dirs)}"
        )

    verifications = {seed: load_verification(path) for seed, path in seed_dirs.items()}
    counts = {seed: load_cross_run(path) for seed, path in seed_dirs.items()}
    for seed in EXPECTED_SEEDS:
        missing = [
            (arm, metric)
            for arm in ARMS
            for metric in METRICS
            if (arm, metric) not in counts[seed]
        ]
        if missing:
            raise SystemExit(f"missing seed {seed} cells: {missing}")
    if ("D0", "peptides") not in counts[2024] or (
        "D0",
        "precursors",
    ) not in counts[2024]:
        raise SystemExit("primary summary is missing D0")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    count_rows: list[dict] = []
    for seed in EXPECTED_SEEDS:
        for arm in ARMS:
            for metric in METRICS:
                count_rows.append(
                    {
                        "carafe_seed": seed,
                        "arm": arm,
                        "metric": metric,
                        "P2_unique": counts[seed][arm, metric]["P2"],
                        "P3_unique": counts[seed][arm, metric]["P3"],
                        "P2_P3_union": counts[seed][arm, metric]["union"],
                    }
                )
    write_tsv(args.output_dir / "carafe_seed_counts.tsv", count_rows)

    contrast_rows: list[dict] = []
    contrast_nets: dict[str, dict[str, dict[str, list[int]]]] = {
        "C3_minus_C2": {metric: {scope: [] for scope in SCOPES} for metric in METRICS},
        "C1_minus_D0": {metric: {scope: [] for scope in SCOPES} for metric in METRICS},
    }
    for seed in EXPECTED_SEEDS:
        for metric in METRICS:
            for scope in SCOPES:
                for label, base_arm, comparison_arm in (
                    ("C3_minus_C2", "C2", "C3"),
                    ("C1_minus_D0", "D0", "C1"),
                ):
                    base_seed = 2024 if base_arm == "D0" else seed
                    base = counts[base_seed][base_arm, metric][scope]
                    comparison = counts[seed][comparison_arm, metric][scope]
                    net = comparison - base
                    contrast_nets[label][metric][scope].append(net)
                    contrast_rows.append(
                        {
                            "carafe_seed": seed,
                            "contrast": label,
                            "scope": scope,
                            "metric": metric,
                            "base_count": base,
                            "comparison_count": comparison,
                            "net": net,
                            "percent_change": f"{100.0 * net / base:.6f}",
                        }
                    )
    write_tsv(args.output_dir / "carafe_seed_contrasts.tsv", contrast_rows)

    interpretation: dict[str, dict] = {}
    for label, metric_data in contrast_nets.items():
        interpretation[label] = {}
        for metric, scope_data in metric_data.items():
            interpretation[label][metric] = {
                scope: {
                    "nets_by_seed": dict(zip(EXPECTED_SEEDS, values)),
                    "signs_by_seed": dict(zip(EXPECTED_SEEDS, map(sign, values))),
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "all_positive": all(value > 0 for value in values),
                    "sign_varies": len({sign(value) for value in values}) > 1,
                }
                for scope, values in scope_data.items()
            }

    source_files = {
        str(path / name): sha256(path / name)
        for path in seed_dirs.values()
        for name in ("cross_run_counts.tsv", "SUMMARY_VERIFICATION.json")
    }
    result = {
        "status": "PASS",
        "seeds": list(EXPECTED_SEEDS),
        "primary_seed": 2024,
        "sensitivity_seeds": [2025, 2026],
        "source_verifications": verifications,
        "interpretation": interpretation,
        "source_sha256": source_files,
    }
    output_json = args.output_dir / "CARAFE_SEED_SENSITIVITY_VERIFICATION.json"
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
