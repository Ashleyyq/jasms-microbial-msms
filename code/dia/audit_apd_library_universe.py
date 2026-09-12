#!/usr/bin/env python3
"""Audit the precursor universe of the two AlphaPeptDeep libraries.

This is a read-only, post-run audit. It compares the stock and epoch-10 APD
exports and then checks whether any library-exclusive precursor was identified
in the corresponding DIA-NN reports after the frozen q-value filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import pyarrow.compute as pc
import pyarrow.parquet as pq


REQUIRED_LIBRARY_COLUMNS = {
    "ModifiedPeptide",
    "PrecursorCharge",
    "RT",
    "StrippedPeptide",
    "PrecursorMz",
    "RelativeIntensity",
}


def normalize_modified_sequence(value: str) -> str:
    value = value.strip()
    if value.startswith("_") and value.endswith("_"):
        value = value[1:-1]
    return value.replace("[UniMod:", "(UniMod:").replace("]", ")")


def precursor_key(modified_sequence: str, charge: str | int) -> tuple[str, int]:
    return normalize_modified_sequence(modified_sequence), int(charge)


def read_library(path: Path) -> tuple[dict[tuple[str, int], dict[str, object]], int]:
    precursors: dict[tuple[str, int], dict[str, object]] = {}
    row_count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"missing header: {path}")
        missing = REQUIRED_LIBRARY_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        for row in reader:
            row_count += 1
            key = precursor_key(row["ModifiedPeptide"], row["PrecursorCharge"])
            intensity = float(row["RelativeIntensity"])
            if key not in precursors:
                precursors[key] = {
                    "modified_sequence": key[0],
                    "charge": key[1],
                    "stripped_sequence": row["StrippedPeptide"],
                    "precursor_mz": float(row["PrecursorMz"]),
                    "rt": float(row["RT"]),
                    "fragment_rows": 0,
                    "positive_fragment_rows": 0,
                    "relative_intensity_sum": 0.0,
                    "relative_intensity_max": -math.inf,
                }
            record = precursors[key]
            record["fragment_rows"] = int(record["fragment_rows"]) + 1
            if intensity > 0:
                record["positive_fragment_rows"] = int(record["positive_fragment_rows"]) + 1
            record["relative_intensity_sum"] = float(record["relative_intensity_sum"]) + intensity
            record["relative_intensity_max"] = max(
                float(record["relative_intensity_max"]), intensity
            )
    return precursors, row_count


def report_keys(path: Path) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    table = pq.read_table(
        path,
        columns=[
            "Modified.Sequence",
            "Precursor.Charge",
            "Q.Value",
            "Global.Q.Value",
        ],
    )
    all_keys = {
        precursor_key(sequence, charge)
        for sequence, charge in zip(
            table["Modified.Sequence"].to_pylist(),
            table["Precursor.Charge"].to_pylist(),
        )
    }
    keep = pc.and_(
        pc.less_equal(table["Q.Value"], 0.01),
        pc.less_equal(table["Global.Q.Value"], 0.01),
    )
    filtered = table.filter(keep)
    sequences = filtered["Modified.Sequence"].to_pylist()
    charges = filtered["Precursor.Charge"].to_pylist()
    accepted_keys = {
        precursor_key(sequence, charge)
        for sequence, charge in zip(sequences, charges)
    }
    return all_keys, accepted_keys


def serialize_record(
    side: str, key: tuple[str, int], record: dict[str, object]
) -> dict[str, object]:
    return {"side": side, **record}


def inspect_hdf_precursor(
    path: Path, record: dict[str, object], min_fragment_mz: float, max_fragment_mz: float
) -> dict[str, object]:
    channels = ("b_z1", "b_z2", "y_z1", "y_z2")
    with h5py.File(path, "r") as handle:
        precursor_group = handle["library/precursor_df"]
        sequence = str(record["stripped_sequence"]).encode()
        charge = int(record["charge"])
        precursor_mz = float(record["precursor_mz"])
        candidates = [
            index
            for index, value in enumerate(precursor_group["sequence"][:])
            if value == sequence
            and int(precursor_group["charge"][index]) == charge
            and abs(float(precursor_group["precursor_mz"][index]) - precursor_mz) <= 1e-6
        ]
        if len(candidates) != 1:
            return {
                "matching_precursor_rows": len(candidates),
                "error": "expected exactly one HDF precursor row",
            }

        index = candidates[0]
        start = int(precursor_group["frag_start_idx"][index])
        stop = int(precursor_group["frag_stop_idx"][index])
        fragment_count = stop - start
        positive_fragments: list[dict[str, object]] = []
        for channel in channels:
            intensities = handle[f"library/fragment_intensity_df/{channel}"][start:stop]
            fragment_mzs = handle[f"library/fragment_mz_df/{channel}"][start:stop]
            series = channel[0]
            for offset, (intensity, fragment_mz) in enumerate(
                zip(intensities, fragment_mzs)
            ):
                if float(intensity) <= 0:
                    continue
                ion_number = offset + 1 if series == "b" else fragment_count - offset
                exportable = min_fragment_mz <= float(fragment_mz) <= max_fragment_mz
                positive_fragments.append(
                    {
                        "ion": f"{series}{ion_number}^{channel[-1]}+",
                        "mz": float(fragment_mz),
                        "relative_intensity": float(intensity),
                        "within_export_mz_range": exportable,
                    }
                )
        return {
            "matching_precursor_rows": 1,
            "hdf_precursor_index": index,
            "raw_positive_fragment_count": len(positive_fragments),
            "positive_fragments_within_export_mz_range": sum(
                bool(fragment["within_export_mz_range"])
                for fragment in positive_fragments
            ),
            "positive_fragments": positive_fragments,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-library", required=True, type=Path)
    parser.add_argument("--epoch10-library", required=True, type=Path)
    parser.add_argument("--stock-hdf", required=True, type=Path)
    parser.add_argument("--epoch10-hdf", required=True, type=Path)
    parser.add_argument("--stock-report", action="append", default=[], type=Path)
    parser.add_argument("--epoch10-report", action="append", default=[], type=Path)
    parser.add_argument("--min-fragment-mz", type=float, default=150.0)
    parser.add_argument("--max-fragment-mz", type=float, default=2000.0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    args = parser.parse_args()

    stock, stock_rows = read_library(args.stock_library)
    epoch10, epoch10_rows = read_library(args.epoch10_library)
    stock_keys = set(stock)
    epoch10_keys = set(epoch10)
    stock_only = sorted(stock_keys - epoch10_keys)
    epoch10_only = sorted(epoch10_keys - stock_keys)
    shared = stock_keys & epoch10_keys

    stock_all_report_hits: dict[tuple[str, int], list[str]] = defaultdict(list)
    epoch10_all_report_hits: dict[tuple[str, int], list[str]] = defaultdict(list)
    stock_accepted_report_hits: dict[tuple[str, int], list[str]] = defaultdict(list)
    epoch10_accepted_report_hits: dict[tuple[str, int], list[str]] = defaultdict(list)
    for report in args.stock_report:
        all_keys, accepted_keys = report_keys(report)
        for key in all_keys.intersection(stock_only):
            stock_all_report_hits[key].append(str(report))
        for key in accepted_keys.intersection(stock_only):
            stock_accepted_report_hits[key].append(str(report))
    for report in args.epoch10_report:
        all_keys, accepted_keys = report_keys(report)
        for key in all_keys.intersection(epoch10_only):
            epoch10_all_report_hits[key].append(str(report))
        for key in accepted_keys.intersection(epoch10_only):
            epoch10_accepted_report_hits[key].append(str(report))

    differences: list[dict[str, object]] = []
    for key in stock_only:
        row = serialize_record("stock_only", key, stock[key])
        row["all_report_hits"] = stock_all_report_hits.get(key, [])
        row["accepted_report_hits"] = stock_accepted_report_hits.get(key, [])
        row["stock_hdf"] = inspect_hdf_precursor(
            args.stock_hdf, row, args.min_fragment_mz, args.max_fragment_mz
        )
        row["epoch10_hdf"] = inspect_hdf_precursor(
            args.epoch10_hdf, row, args.min_fragment_mz, args.max_fragment_mz
        )
        differences.append(row)
    for key in epoch10_only:
        row = serialize_record("epoch10_only", key, epoch10[key])
        row["all_report_hits"] = epoch10_all_report_hits.get(key, [])
        row["accepted_report_hits"] = epoch10_accepted_report_hits.get(key, [])
        row["stock_hdf"] = inspect_hdf_precursor(
            args.stock_hdf, row, args.min_fragment_mz, args.max_fragment_mz
        )
        row["epoch10_hdf"] = inspect_hdf_precursor(
            args.epoch10_hdf, row, args.min_fragment_mz, args.max_fragment_mz
        )
        differences.append(row)

    rt_differences = [abs(float(stock[k]["rt"]) - float(epoch10[k]["rt"])) for k in shared]
    mz_differences = [
        abs(float(stock[k]["precursor_mz"]) - float(epoch10[k]["precursor_mz"]))
        for k in shared
    ]
    shared_fragment_count_differences = sum(
        int(stock[k]["fragment_rows"]) != int(epoch10[k]["fragment_rows"])
        for k in shared
    )

    result = {
        "status": (
            "PASS"
            if not stock_all_report_hits and not epoch10_all_report_hits
            else "REVIEW"
        ),
        "interpretation": (
            "Library-exclusive precursors did not enter any report."
            if not stock_all_report_hits and not epoch10_all_report_hits
            else "At least one library-exclusive precursor entered a report."
        ),
        "key_definition": ["normalized modified sequence", "precursor charge"],
        "q_value_filter": "Q.Value <= 0.01 and Global.Q.Value <= 0.01",
        "fragment_mz_export_range": [args.min_fragment_mz, args.max_fragment_mz],
        "stock": {"fragment_rows": stock_rows, "precursors": len(stock)},
        "epoch10": {"fragment_rows": epoch10_rows, "precursors": len(epoch10)},
        "comparison": {
            "shared_precursors": len(shared),
            "stock_only_precursors": len(stock_only),
            "epoch10_only_precursors": len(epoch10_only),
            "shared_with_different_fragment_row_count": shared_fragment_count_differences,
            "max_absolute_rt_difference": max(rt_differences, default=0.0),
            "max_absolute_precursor_mz_difference": max(mz_differences, default=0.0),
            "library_exclusive_precursors_with_accepted_report_hits": sum(
                bool(row["accepted_report_hits"]) for row in differences
            ),
            "library_exclusive_precursors_with_any_report_hits": sum(
                bool(row["all_report_hits"]) for row in differences
            ),
        },
        "differences": differences,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = [
        "side",
        "modified_sequence",
        "charge",
        "stripped_sequence",
        "precursor_mz",
        "rt",
        "fragment_rows",
        "positive_fragment_rows",
        "relative_intensity_sum",
        "relative_intensity_max",
        "all_report_hits",
        "accepted_report_hits",
    ]
    with args.output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in differences:
            output = dict(row)
            output["all_report_hits"] = ";".join(output["all_report_hits"])
            output["accepted_report_hits"] = ";".join(output["accepted_report_hits"])
            output.pop("stock_hdf")
            output.pop("epoch10_hdf")
            writer.writerow(output)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
