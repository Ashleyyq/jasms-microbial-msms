#!/usr/bin/env python3
"""Rebuild all eligible PXD010613 annotated spectra without a scan-order cap.

This script deliberately imports the already-audited conversion primitives from
``convert_mzml_mzid_to_mgf.py`` on the cluster.  It changes only the output
policy: every rank-1 PSM with MS-GF+ QValue <= the requested cutoff and a
supported modification is written when its MS2 scan is present in mzML.

The output directory must not exist.  A machine-readable manifest records all
input/output hashes and denominators.  The script refuses unexpected source
composition, missing scans, duplicate species mappings, and count drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import convert_mzml_mzid_to_mgf as legacy


EXPECTED_SPECIES = {
    "Akkermansia_muciniphila": 26604,
    "Caulobacter_crescentus": 23556,
    "Enterococcus_faecalis": 18986,
    "Halanaerobium_congolense": 34756,
}
EXPECTED_TOTAL = sum(EXPECTED_SPECIES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uncapped, non-overwriting PXD010613 mzML/mzID to MGF rebuild"
    )
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--reference_dir", required=True,
        help="Directory with the historical 5,000-entry MGF files for prefix regression",
    )
    parser.add_argument("--fdr", type=float, default=0.01)
    parser.add_argument(
        "--expected_total", type=int, default=EXPECTED_TOTAL,
        help="Fail on count drift; set only after auditing a changed source set",
    )
    args = parser.parse_args()
    if not 0 < args.fdr <= 1:
        parser.error("--fdr must be in (0, 1]")
    if args.expected_total < 1:
        parser.error("--expected_total must be positive")
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


def count_mgf_entries(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip() == "BEGIN IONS")


def is_binary_prefix(prefix_path: Path, full_path: Path, block_size: int = 1024 * 1024) -> bool:
    with prefix_path.open("rb") as prefix, full_path.open("rb") as full:
        while True:
            expected = prefix.read(block_size)
            if not expected:
                return True
            if full.read(len(expected)) != expected:
                return False


def serializable_stats(stats: dict) -> dict:
    return {
        "total_psms": int(stats["total_psms"]),
        "rank1_psms": int(stats["rank1_psms"]),
        "qvalue_passed": int(stats["fdr_passed"]),
        "supported_mod_psms": int(stats["supported_mods"]),
        "unsupported_mod_psms": int(stats["unsupported_mods"]),
        "no_scan_identifier": int(stats["no_scan"]),
        "unique_unmodified_sequences": int(stats["n_unique"]),
        "scans_with_psm": int(stats["n_scans_with_psm"]),
        "spectrum_id_formats": dict(stats["spectrum_id_formats"]),
        "modification_distribution": dict(stats["mod_distribution"]),
        "unsupported_mod_details": dict(stats["unsupported_mod_details"]),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_psm_index(path: Path, psms: dict) -> None:
    """Map sequential MGF entry numbers back to raw scan identifiers."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["mgf_entry", "raw_scan", "sequence", "modified_sequence", "charge", "qvalue"]
        )
        for mgf_entry, raw_scan in enumerate(sorted(psms), start=1):
            psm = psms[raw_scan]
            writer.writerow(
                [
                    mgf_entry,
                    raw_scan,
                    psm["sequence"],
                    psm["seq_inline"],
                    psm["charge"],
                    format(psm["qvalue"], ".12g"),
                ]
            )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"ERROR: raw directory not found: {raw_dir}")
    if not reference_dir.is_dir():
        raise SystemExit(f"ERROR: reference directory not found: {reference_dir}")
    if out_dir.exists():
        raise SystemExit(f"ERROR: refusing existing output directory: {out_dir}")

    mzid_files = sorted(raw_dir.glob("*.mzid.gz"))
    mzml_files = sorted(raw_dir.glob("*.mzML.gz"))
    if len(mzid_files) != 4 or len(mzml_files) != 4:
        raise SystemExit(
            f"ERROR: expected four mzID and four mzML files; found "
            f"{len(mzid_files)} and {len(mzml_files)}"
        )

    species_inputs: dict[str, tuple[Path, Path]] = {}
    for mzid_path in mzid_files:
        base = mzid_path.name.removesuffix("_msgfplus.mzid.gz")
        species = legacy.filename_to_species(base)
        if species is None:
            raise SystemExit(f"ERROR: unmapped mzID file: {mzid_path.name}")
        if species in species_inputs:
            raise SystemExit(f"ERROR: multiple mzID inputs map to {species}")
        mzml_path = raw_dir / f"{base}.mzML.gz"
        if not mzml_path.is_file():
            raise SystemExit(f"ERROR: paired mzML not found: {mzml_path}")
        species_inputs[species] = (mzid_path, mzml_path)

    if set(species_inputs) != set(EXPECTED_SPECIES):
        raise SystemExit(
            f"ERROR: species mismatch: {sorted(species_inputs)} != "
            f"{sorted(EXPECTED_SPECIES)}"
        )

    out_dir.mkdir(parents=False)
    manifest: dict = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PXD010613",
        "selection": {
            "psm_rank": 1,
            "msgf_qvalue_operator": "<=",
            "msgf_qvalue_cutoff": args.fdr,
            "scan_order_cap": None,
            "supported_modifications": sorted(legacy.SUPPORTED_MODS),
        },
        "expected_total": args.expected_total,
        "legacy_converter": {
            "path": str(Path(legacy.__file__).resolve()),
            "sha256": sha256_file(Path(legacy.__file__).resolve()),
        },
        "species": {},
    }

    total_written = 0
    for species in sorted(species_inputs):
        mzid_path, mzml_path = species_inputs[species]
        print(f"[{species}] parsing {mzid_path.name}", flush=True)
        psms, stats = legacy.parse_mzid_file(str(mzid_path), args.fdr)
        expected = EXPECTED_SPECIES[species]
        if len(psms) != expected:
            raise RuntimeError(
                f"{species}: source count drift, got {len(psms)}, expected {expected}"
            )

        print(f"[{species}] extracting {len(psms)} MS2 scans", flush=True)
        spectra = legacy.parse_mzml_spectra(str(mzml_path), set(psms))
        if len(spectra) != len(psms):
            missing = sorted(set(psms) - set(spectra))
            raise RuntimeError(
                f"{species}: {len(missing)} PSM scans absent from mzML; "
                f"first={missing[:10]}"
            )

        output_mgf = out_dir / f"{species}.mgf"
        written, missing_count = legacy.write_mgf(
            psms, spectra, str(output_mgf), max_spectra=len(psms)
        )
        if written != len(psms) or missing_count != 0:
            raise RuntimeError(
                f"{species}: write mismatch, written={written}, "
                f"expected={len(psms)}, missing={missing_count}"
            )
        psm_index = out_dir / f"{species}.psm_index.tsv"
        write_psm_index(psm_index, psms)
        reference_mgf = reference_dir / f"{species}.mgf"
        if not reference_mgf.is_file():
            raise RuntimeError(f"{species}: historical reference MGF not found")
        reference_entries = count_mgf_entries(reference_mgf)
        if reference_entries != 5000:
            raise RuntimeError(
                f"{species}: historical reference has {reference_entries} entries, expected 5000"
            )
        if not is_binary_prefix(reference_mgf, output_mgf):
            raise RuntimeError(
                f"{species}: uncapped MGF does not preserve the historical 5,000-entry prefix"
            )

        manifest["species"][species] = {
            "mzid": {
                "path": str(mzid_path),
                "size_bytes": mzid_path.stat().st_size,
                "sha256": sha256_file(mzid_path),
            },
            "mzml": {
                "path": str(mzml_path),
                "size_bytes": mzml_path.stat().st_size,
                "sha256": sha256_file(mzml_path),
            },
            "parse": serializable_stats(stats),
            "scan_number_min": min(psms),
            "scan_number_max": max(psms),
            "mgf": {
                "path": str(output_mgf),
                "entries": written,
                "size_bytes": output_mgf.stat().st_size,
                "sha256": sha256_file(output_mgf),
            },
            "psm_index": {
                "path": str(psm_index),
                "rows": written,
                "size_bytes": psm_index.stat().st_size,
                "sha256": sha256_file(psm_index),
            },
            "historical_prefix_regression": {
                "reference_path": str(reference_mgf),
                "reference_entries": reference_entries,
                "reference_sha256": sha256_file(reference_mgf),
                "binary_prefix_match": True,
            },
        }
        total_written += written
        print(f"[{species}] wrote {written} spectra to {output_mgf}", flush=True)

    if total_written != args.expected_total:
        raise RuntimeError(
            f"total write mismatch: {total_written} != {args.expected_total}"
        )
    manifest["total_mgf_entries"] = total_written
    atomic_json(out_dir / "conversion_manifest.json", manifest)
    print(f"Completed uncapped PXD010613 rebuild: {total_written} spectra", flush=True)
    print(f"Output directory: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
