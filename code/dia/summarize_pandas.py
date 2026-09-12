#!/usr/bin/env python3
"""Primary pandas implementation of the final P2/P3 DIA summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ARMS = ("D0", "D1", "D2", "D3", "C1", "C2", "C3")
RUNS = ("P2", "P3")
CONTRASTS = (("D3_minus_D2", "D2", "D3"), ("C3_minus_C2", "C2", "C3"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_path(root: Path, arm: str, run: str) -> Path:
    layer = "direct" if arm.startswith("D") else "carafe_search"
    return root / layer / arm / f"test_{run}" / "report.parquet"


def decoy_mask(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "Decoy" not in frame.columns:
        return pd.Series(False, index=frame.index), "DIA-NN main report has no Decoy column"
    values = frame["Decoy"]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False), "Decoy == True excluded"
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).ne(0), "nonzero Decoy excluded"
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "decoy"}), "truthy Decoy excluded"


def identifiers(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].dropna().astype(str)
    return {value for value in values if value and value.lower() != "nan"}


def read_report(path: Path) -> tuple[dict, dict[str, set[str]]]:
    if not path.is_file():
        raise SystemExit(f"missing report: {path}")
    frame = pd.read_parquet(path)
    required = {"Q.Value", "Global.Q.Value", "Precursor.Id", "Stripped.Sequence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"missing required columns in {path}: {missing}")

    is_decoy, decoy_rule = decoy_mask(frame)
    targets = frame.loc[~is_decoy]
    kept = targets.loc[
        targets["Q.Value"].le(0.01) & targets["Global.Q.Value"].le(0.01)
    ]
    sets = {
        "precursors": identifiers(kept, "Precursor.Id"),
        "peptides": identifiers(kept, "Stripped.Sequence"),
    }
    stats = {
        "report_rows": int(len(frame)),
        "target_rows": int(len(targets)),
        "rows_1pct": int(len(kept)),
        "unique_precursors": len(sets["precursors"]),
        "unique_peptides": len(sets["peptides"]),
        "decoy_rule": decoy_rule,
        "report_sha256": sha256(path),
    }
    return stats, sets


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite: {path}")
    frame.to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    per_run_rows: list[dict] = []
    all_sets: dict[str, dict[str, dict[str, set[str]]]] = {}
    for arm in ARMS:
        all_sets[arm] = {}
        for run in RUNS:
            path = report_path(args.run_root, arm, run)
            stats, sets = read_report(path)
            all_sets[arm][run] = sets
            per_run_rows.append(
                {
                    "arm": arm,
                    "run": run,
                    "report": str(path),
                    **stats,
                }
            )

    per_run = pd.DataFrame(per_run_rows).sort_values(["arm", "run"])
    write_tsv(per_run, output / "per_run_counts.tsv")

    cross_rows: list[dict] = []
    union_sets: dict[str, dict[str, set[str]]] = {}
    for arm in ARMS:
        union_sets[arm] = {}
        for metric in ("peptides", "precursors"):
            p2 = all_sets[arm]["P2"][metric]
            p3 = all_sets[arm]["P3"][metric]
            union = p2 | p3
            union_sets[arm][metric] = union
            cross_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "P2_unique": len(p2),
                    "P3_unique": len(p3),
                    "run_summed": len(p2) + len(p3),
                    "P2_P3_shared": len(p2 & p3),
                    "P2_P3_union": len(union),
                }
            )
    cross = pd.DataFrame(cross_rows).sort_values(["arm", "metric"])
    write_tsv(cross, output / "cross_run_counts.tsv")

    contrast_rows: list[dict] = []
    membership_rows: list[dict] = []
    interpretation: dict[str, dict] = {}
    for label, base_arm, comparison_arm in CONTRASTS:
        interpretation[label] = {}
        for metric in ("peptides", "precursors"):
            signs: dict[str, int] = {}
            for scope in (*RUNS, "union"):
                if scope == "union":
                    base = union_sets[base_arm][metric]
                    comparison = union_sets[comparison_arm][metric]
                else:
                    base = all_sets[base_arm][scope][metric]
                    comparison = all_sets[comparison_arm][scope][metric]
                gained = comparison - base
                lost = base - comparison
                net = len(comparison) - len(base)
                signs[scope] = 1 if net > 0 else (-1 if net < 0 else 0)
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
                        "gained": len(gained),
                        "lost": len(lost),
                        "net": net,
                    }
                )
                if scope == "union":
                    for direction, values in (("gained", gained), ("lost", lost)):
                        membership_rows.extend(
                            {
                                "contrast": label,
                                "metric": metric,
                                "direction": direction,
                                "identifier": value,
                            }
                            for value in sorted(values)
                        )
            interpretation[label][metric] = {
                "direction": signs,
                "positive_in_both_runs_and_union": all(
                    signs[scope] > 0 for scope in (*RUNS, "union")
                ),
                "mixed_between_runs": signs["P2"] * signs["P3"] < 0,
            }

    contrasts = pd.DataFrame(contrast_rows).sort_values(
        ["contrast", "metric", "scope"]
    )
    write_tsv(contrasts, output / "paired_contrasts.tsv")
    membership = pd.DataFrame(
        membership_rows, columns=["contrast", "metric", "direction", "identifier"]
    )
    write_tsv(membership, output / "union_membership_changes.tsv")

    manifest = {
        "design": {
            "training_run": "P1 for C1-C3 only",
            "primary_evaluation_runs": list(RUNS),
            "replicate_scope": "technical runs; no biological-replicate claim",
            "primary_endpoint": "P2/P3 union of unique Stripped.Sequence",
            "primary_contrasts": [item[0] for item in CONTRASTS],
            "filter": "target rows with Q.Value <= 0.01 and Global.Q.Value <= 0.01",
        },
        "interpretation_flags": interpretation,
        "files": {},
    }
    for name in (
        "per_run_counts.tsv",
        "cross_run_counts.tsv",
        "paired_contrasts.tsv",
        "union_membership_changes.tsv",
    ):
        path = output / name
        manifest["files"][name] = {"path": str(path), "sha256": sha256(path)}
    manifest_path = output / "primary_summary_manifest.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
