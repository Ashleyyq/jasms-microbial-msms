"""
convert_mzml_mzid_to_mgf.py
============================
Convert PXD010613 mzML + mzid (MS-GF+ identifications) → annotated MGF files
compatible with run_all.py's parse_mgf_file() / convert_mod().

Produces one MGF per organism with SEQ= annotations in MassIVE inline format
(e.g. PEPTM+16.0IDEK, not Oxidation@M).

Features:
  - --dry_run mode: parse mzid only, report PSM counts / spectrumID format /
    modification distribution / expected output. Does NOT touch mzML or write MGF.
  - Contamination audit: compare PXD010613 sequences against ProteomeTools
    training peptides (MSP files on cluster) to quantify overlap.
  - Explicit handling of unsupported modifications (skip + log count).
  - Supports both spectrumID formats: "scan=NNNN" and "index=NNN".

Usage (dry run — mzid parsing only):
  python scripts/convert_mzml_mzid_to_mgf.py \\
    --raw_dir data/bacteria_hcd/raw \\
    --out_dir data/bacteria_hcd \\
    --fdr 0.01 --max_spectra 5000 \\
    --training_msp_dir data/proteometools \\
    --dry_run

Usage (full conversion):
  python scripts/convert_mzml_mzid_to_mgf.py \\
    --raw_dir data/bacteria_hcd/raw \\
    --out_dir data/bacteria_hcd \\
    --fdr 0.01 --max_spectra 5000 \\
    --training_msp_dir data/proteometools
"""

import argparse
import gzip
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _decompress_gz(gz_path: str) -> str:
    """Decompress a .gz file to a temp file if needed.  Returns path to use.
    If the file is not gzipped, returns the original path.
    Caller must delete the temp file when done (returned path differs from input).
    """
    if not gz_path.endswith('.gz'):
        return gz_path
    # Decompress to a named temp file (same suffix minus .gz)
    suffix = os.path.splitext(gz_path[:-3])[1]  # e.g. '.mzid' or '.mzML'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with gzip.open(gz_path, 'rb') as f_in, open(tmp_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return tmp_path

# ── Species mapping ──────────────────────────────────────────────────────────

SPECIES_MAP = {
    'Alverdy_Efae':              'Enterococcus_faecalis',
    'Biodiversity_A_muciniphila': 'Akkermansia_muciniphila',
    'Ha_':                        'Halanaerobium_congolense',
    'YJ_Cc':                      'Caulobacter_crescentus',
}


def filename_to_species(fname: str) -> str:
    """Map an mzid/mzML filename to a species name via prefix matching."""
    for prefix, species in SPECIES_MAP.items():
        if fname.startswith(prefix):
            return species
    return None


# ── Modification handling ────────────────────────────────────────────────────

# Supported modifications: mapped to MassIVE-style inline notation
# Format in SEQ= field: "M+16.0" not "Oxidation@M"
SUPPORTED_MODS = {
    'Oxidation':        ('+16.0',  {'M'}),
    'Carbamidomethyl':  ('+57.0',  {'C'}),
}

# Track unsupported mods encountered
_UNSUPPORTED_MOD_COUNTER = Counter()


def build_seq_with_inline_mods(sequence: str, modifications: list) -> str:
    """
    Build a SEQ= string with MassIVE-style inline modifications.

    Args:
        sequence: bare peptide sequence (uppercase AA letters)
        modifications: list of dicts from pyteomics mzid, each with keys:
            'location' (int, 1-based; 0 = N-term),
            'name' or 'accession',
            'residues' (list of str, optional)

    Returns:
        Modified sequence string (e.g. "PEPTM+16.0IDEK") or None if
        unsupported modifications were found (spectrum should be skipped).
    """
    if not modifications:
        return sequence

    # Collect inline insertions: position → mass_delta_str
    inserts = {}  # key = position in sequence (0-based after AA), value = "+NN.N"
    has_unsupported = False

    for mod in modifications:
        loc = mod.get('location', -1)
        # Try multiple keys for the modification name
        mod_name = mod.get('name', '')
        if not mod_name:
            mod_name = mod.get('accession', '')

        # Check if this is a supported modification
        matched = False
        for sup_name, (delta_str, valid_residues) in SUPPORTED_MODS.items():
            if sup_name.lower() in mod_name.lower():
                # location: 0 = N-term, 1..n = residue position (1-based)
                if loc < 1 or loc > len(sequence):
                    # N-term or C-term mod — treat as unsupported for now
                    _UNSUPPORTED_MOD_COUNTER[f"{mod_name}(terminal)"] += 1
                    has_unsupported = True
                    break
                aa = sequence[loc - 1]
                if aa not in valid_residues:
                    _UNSUPPORTED_MOD_COUNTER[f"{mod_name}@{aa}(wrong_residue)"] += 1
                    has_unsupported = True
                    break
                inserts[loc - 1] = delta_str
                matched = True
                break

        if not matched and not has_unsupported:
            # Unsupported modification — skip this spectrum
            _UNSUPPORTED_MOD_COUNTER[mod_name if mod_name else 'Unknown'] += 1
            has_unsupported = True

    if has_unsupported:
        return None

    # Build the inline sequence
    parts = []
    for i, aa in enumerate(sequence):
        parts.append(aa)
        if i in inserts:
            parts.append(inserts[i])
    return ''.join(parts)


# ── spectrumID parsing ───────────────────────────────────────────────────────

RE_SCAN = re.compile(r'scan=(\d+)')
RE_INDEX = re.compile(r'index=(\d+)')


def parse_spectrum_id(spectrum_id: str):
    """
    Extract scan number from spectrumID string.
    Supports:
      - "controllerType=0 controllerNumber=1 scan=12345"
      - "index=456"
    Returns (scan_number: int, format_str: str) or (None, None).
    """
    m = RE_SCAN.search(spectrum_id)
    if m:
        return int(m.group(1)), 'scan=NNNN'
    m = RE_INDEX.search(spectrum_id)
    if m:
        return int(m.group(1)), 'index=NNN'
    return None, None


# ── mzid parsing ─────────────────────────────────────────────────────────────

def parse_mzid_file(mzid_path: str, fdr_cutoff: float = 0.01):
    """
    Parse an mzid.gz file and extract filtered PSMs.

    Returns:
        psms: dict mapping scan_number → {
            'sequence': str (bare),
            'seq_inline': str (with inline mods) or None,
            'charge': int,
            'qvalue': float,
        }
        stats: dict with parsing statistics
    """
    from pyteomics import mzid

    stats = {
        'total_psms': 0,
        'rank1_psms': 0,
        'fdr_passed': 0,
        'supported_mods': 0,
        'unsupported_mods': 0,
        'no_scan': 0,
        'unique_sequences': set(),
        'spectrum_id_formats': Counter(),
        'mod_distribution': Counter(),
    }

    # Reset unsupported mod counter for this file
    _UNSUPPORTED_MOD_COUNTER.clear()

    psms = {}

    # Decompress .gz if needed — pyteomics may not handle gzipped mzid
    decompressed = _decompress_gz(mzid_path)
    try:
        with mzid.MzIdentML(decompressed) as reader:
            for spectrum_result in reader:
                spec_id = spectrum_result.get('spectrumID', '')
                scan_num, id_format = parse_spectrum_id(spec_id)

                if scan_num is None:
                    stats['no_scan'] += 1
                    continue
                stats['spectrum_id_formats'][id_format] += 1

                # Get the best-ranking identification for this spectrum
                items = spectrum_result.get('SpectrumIdentificationItem', [])
                for item in items:
                    stats['total_psms'] += 1

                    # Rank filter
                    rank = item.get('rank', 1)
                    if rank != 1:
                        continue
                    stats['rank1_psms'] += 1

                    # FDR filter — try multiple score names
                    qvalue = None
                    for key in ('MS-GF:QValue', 'QValue', 'MS-GF:PepQValue',
                                'q-value', 'MS-GF:EValue'):
                        if key in item:
                            qvalue = float(item[key])
                            break
                    if qvalue is None:
                        # Try params
                        for param_key in item:
                            if 'qvalue' in param_key.lower() or 'q-value' in param_key.lower():
                                qvalue = float(item[param_key])
                                break

                    if qvalue is None or qvalue > fdr_cutoff:
                        continue
                    stats['fdr_passed'] += 1

                    # Extract peptide sequence
                    pep_seq = item.get('PeptideSequence', '')
                    if not pep_seq:
                        continue

                    # Extract charge
                    charge = item.get('chargeState', 2)

                    # Extract modifications
                    modifications = item.get('Modification', [])
                    for mod in modifications:
                        mod_name = mod.get('name', mod.get('accession', 'Unknown'))
                        stats['mod_distribution'][mod_name] += 1

                    # Build inline sequence
                    seq_inline = build_seq_with_inline_mods(pep_seq, modifications)
                    if seq_inline is None:
                        stats['unsupported_mods'] += 1
                        continue
                    stats['supported_mods'] += 1

                    stats['unique_sequences'].add(pep_seq)

                    # Keep best Q-value per scan
                    if scan_num not in psms or qvalue < psms[scan_num]['qvalue']:
                        psms[scan_num] = {
                            'sequence': pep_seq,
                            'seq_inline': seq_inline,
                            'charge': int(charge),
                            'qvalue': float(qvalue),
                        }
    finally:
        if decompressed != mzid_path:
            os.unlink(decompressed)

    stats['unsupported_mod_details'] = dict(_UNSUPPORTED_MOD_COUNTER)
    stats['n_unique'] = len(stats['unique_sequences'])
    stats['n_scans_with_psm'] = len(psms)
    return psms, stats


# ── mzML parsing ─────────────────────────────────────────────────────────────

def parse_mzml_spectra(mzml_path: str, scan_numbers: set):
    """
    Parse an mzML.gz file and extract MS2 spectra for the given scan numbers.

    Returns:
        spectra: dict mapping scan_number → {
            'mz': np.array,
            'intensity': np.array,
            'precursor_mz': float,
        }
    """
    from pyteomics import mzml

    spectra = {}
    found = 0
    total = 0

    # Decompress .gz if needed — pyteomics may not handle gzipped mzML
    decompressed = _decompress_gz(mzml_path)
    try:
        with mzml.MzML(decompressed) as reader:
            for spectrum in reader:
                total += 1
                # Extract scan number from the spectrum ID
                spec_id = spectrum.get('id', '')
                scan_num, _ = parse_spectrum_id(spec_id)

                if scan_num is None:
                    # Try the 'index' attribute
                    scan_num = spectrum.get('index')

                if scan_num not in scan_numbers:
                    continue

                # Check this is an MS2 spectrum
                ms_level = spectrum.get('ms level', 0)
                if ms_level != 2:
                    continue

                # Extract m/z and intensity arrays
                mz_array = spectrum.get('m/z array')
                int_array = spectrum.get('intensity array')
                if mz_array is None or int_array is None:
                    continue

                # Extract precursor m/z
                precursor_mz = None
                precursors = spectrum.get('precursorList', {}).get('precursor', [])
                if not precursors:
                    # pyteomics may store precursor info differently
                    selected = spectrum.get('selectedIonList', {})
                    if selected:
                        ions = selected.get('selectedIon', [])
                        if ions:
                            precursor_mz = ions[0].get('selected ion m/z')
                else:
                    sel_ion = precursors[0].get('selectedIonList', {}).get('selectedIon', [])
                    if sel_ion:
                        precursor_mz = sel_ion[0].get('selected ion m/z')

                # pyteomics often puts precursor info at top level
                if precursor_mz is None:
                    prec_list = spectrum.get('precursorList', [])
                    if isinstance(prec_list, list) and prec_list:
                        precursor_mz = prec_list[0].get('selected ion m/z')

                if precursor_mz is None:
                    # Last resort: check if precursor info is in a flatter structure
                    for key in spectrum:
                        if 'selected ion m/z' in str(key):
                            precursor_mz = spectrum[key]
                            break

                if precursor_mz is None:
                    continue

                spectra[scan_num] = {
                    'mz': np.array(mz_array, dtype=np.float64),
                    'intensity': np.array(int_array, dtype=np.float64),
                    'precursor_mz': float(precursor_mz),
                }
                found += 1

                if found % 2000 == 0:
                    print(f"    Extracted {found}/{len(scan_numbers)} target scans "
                          f"(read {total} spectra so far)...", flush=True)

                if found >= len(scan_numbers):
                    break
    finally:
        if decompressed != mzml_path:
            os.unlink(decompressed)

    print(f"    Extracted {found}/{len(scan_numbers)} target scans "
          f"from {total} total spectra", flush=True)
    return spectra


# ── MGF writing ──────────────────────────────────────────────────────────────

def write_mgf(psms: dict, spectra: dict, out_path: str, max_spectra: int):
    """
    Write annotated MGF file matching parse_mgf_file() format.
    PEPMASS = precursor m/z (NOT neutral mass).
    """
    written = 0
    missing_spectrum = 0

    with open(out_path, 'w') as f:
        for scan_num in sorted(psms.keys()):
            if written >= max_spectra:
                break
            if scan_num not in spectra:
                missing_spectrum += 1
                continue

            psm = psms[scan_num]
            spec = spectra[scan_num]

            f.write("BEGIN IONS\n")
            f.write(f"PEPMASS={spec['precursor_mz']:.6f}\n")
            f.write(f"CHARGE={psm['charge']}+\n")
            f.write(f"SEQ={psm['seq_inline']}\n")

            mz = spec['mz']
            intensity = spec['intensity']
            for m, i in zip(mz, intensity):
                f.write(f"{m:.6f} {i:.4f}\n")

            f.write("END IONS\n\n")
            written += 1

    return written, missing_spectrum


# ── ProteomeTools MSP sequence extraction ────────────────────────────────────

def load_proteometools_sequences(msp_dir: str) -> set:
    """
    Extract unique stripped peptide sequences from ProteomeTools MSP files.
    Parses 'Name:' lines which have format like "AAAPSVTLFPPSSEELQANK/2"
    (sequence/charge). Strips modifications and returns uppercase sequences.
    """
    sequences = set()
    msp_files = sorted(Path(msp_dir).glob('FTMS_HCD_*_annotated_*.msp'))
    if not msp_files:
        print(f"  WARNING: No FTMS_HCD MSP files found in {msp_dir}")
        return sequences

    for msp_path in msp_files:
        n_before = len(sequences)
        with open(msp_path, 'r') as f:
            for line in f:
                if line.startswith('Name: '):
                    # Format: "Name: AAAPSVTLFPPSSEELQANK/2"
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name = parts[1]
                        # Strip charge suffix
                        seq = name.split('/')[0]
                        # Strip modification annotations if any
                        # (some MSP files have modified names like "AAAM(ox)IDEK")
                        clean = re.sub(r'\([^)]*\)', '', seq).upper()
                        # Keep only standard AA
                        if clean and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in clean):
                            sequences.add(clean)
        n_new = len(sequences) - n_before
        print(f"  {msp_path.name}: +{n_new} unique sequences "
              f"(total: {len(sequences)})", flush=True)

    return sequences


# ── Contamination audit ──────────────────────────────────────────────────────

def run_contamination_audit(pxd_sequences: set, training_msp_dir: str):
    """
    Compare PXD010613 peptide sequences against ProteomeTools training set.
    Prints report to stdout.
    Returns overlap percentage.
    """
    print("\n" + "=" * 70)
    print("CONTAMINATION AUDIT")
    print("=" * 70)

    print(f"\nPXD010613 organisms:")
    for species in sorted(SPECIES_MAP.values()):
        print(f"  - {species.replace('_', ' ')}")
    print(f"PXD010613 unique peptide sequences: {len(pxd_sequences)}")

    # Species-level audit
    print(f"\nvs Prosit (ProteomeTools training, Part I):")
    print(f"  Species overlap: NONE (synthetic human vs bacterial)")

    # Sequence-level audit
    if training_msp_dir:
        print(f"\n  Loading ProteomeTools sequences from {training_msp_dir} ...")
        pt_sequences = load_proteometools_sequences(training_msp_dir)
        if pt_sequences:
            overlap = pxd_sequences & pt_sequences
            pct = len(overlap) / len(pxd_sequences) * 100 if pxd_sequences else 0
            print(f"\n  ProteomeTools unique sequences loaded: {len(pt_sequences)}")
            print(f"  Sequence-level overlap: {len(overlap)} / {len(pxd_sequences)} "
                  f"({pct:.1f}%)")
            if overlap:
                examples = sorted(overlap)[:5]
                print(f"  Examples: {', '.join(examples)}")
        else:
            pct = -1
            print(f"  Sequence-level overlap: could not compute (no MSP sequences loaded)")
    else:
        pct = -1
        print(f"  Sequence-level overlap: not audited (--training_msp_dir not provided)")

    print(f"\nvs APD_ms2_generic training set (17 PXDs, 4 organisms):")
    print(f"  Species overlap: NONE (E. coli, human, yeast, drosophila — no match)")
    print(f"  Sequence-level overlap: not audited (17 PXD training sets infeasible;")
    print(f"    rely on species-level non-overlap + conservation argument)")

    # Decision
    print()
    if pct > 30:
        print(f"  *** DECISION: STOP — overlap {pct:.1f}% exceeds 30% threshold ***")
        return pct
    elif pct > 5:
        print(f"  *** DECISION: STRATIFY — overlap {pct:.1f}% is 5-30%; "
              f"add --stratify flag to eval ***")
    elif pct >= 0:
        print(f"  *** DECISION: PROCEED — overlap {pct:.1f}% is negligible ***")
    else:
        print(f"  *** DECISION: PROCEED (sequence audit not performed; "
              f"species-level shows ZERO overlap) ***")

    print("=" * 70)
    return pct


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Convert PXD010613 mzML + mzid → annotated MGF for HCD bacterial evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--raw_dir', required=True,
                    help='Directory containing .mzML.gz and .mzid.gz files')
    ap.add_argument('--out_dir', required=True,
                    help='Output directory for per-species .mgf files')
    ap.add_argument('--fdr', type=float, default=0.01,
                    help='MS-GF+ Q-value cutoff (default: 0.01 = 1%% FDR)')
    ap.add_argument('--max_spectra', type=int, default=5000,
                    help='Max spectra per species in output MGF (default: 5000)')
    ap.add_argument('--training_msp_dir', default=None,
                    help='Path to ProteomeTools MSP files for contamination audit')
    ap.add_argument('--dry_run', action='store_true',
                    help='Parse mzid only — report stats, do NOT write MGF or parse mzML')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find mzid files
    mzid_files = sorted(raw_dir.glob('*.mzid.gz'))
    if not mzid_files:
        sys.exit(f"ERROR: No .mzid.gz files found in {raw_dir}")

    print(f"Found {len(mzid_files)} mzid.gz files in {raw_dir}")
    if args.dry_run:
        print("\n*** DRY RUN — mzid parsing only, no mzML or MGF output ***\n")
    print(f"FDR cutoff: {args.fdr}")
    print(f"Max spectra per species: {args.max_spectra}")
    print()

    # ── Process each mzid file ───────────────────────────────────────────────

    all_sequences = set()
    species_psms = defaultdict(dict)   # species_name → {scan → psm}
    species_stats = {}

    for mzid_path in mzid_files:
        fname = mzid_path.name
        species = filename_to_species(fname.replace('_msgfplus.mzid.gz', ''))
        if species is None:
            print(f"WARNING: Cannot map {fname} to a species — skipping")
            continue

        print(f"{'='*60}")
        print(f"File: {fname}")
        print(f"Species: {species.replace('_', ' ')}")
        print(f"{'='*60}")

        psms, stats = parse_mzid_file(str(mzid_path), args.fdr)

        # Report
        formats_str = ', '.join(f"{fmt} ({n})" for fmt, n in
                                stats['spectrum_id_formats'].most_common())
        print(f"  spectrumID format: {formats_str or 'NONE DETECTED'}")
        print(f"  Total PSMs: {stats['total_psms']}")
        print(f"  Rank-1 PSMs: {stats['rank1_psms']}")
        print(f"  FDR < {args.fdr} (rank 1): {stats['fdr_passed']}")
        print(f"  Supported mods (kept): {stats['supported_mods']}")
        print(f"  Unsupported mods (skipped): {stats['unsupported_mods']}")
        print(f"  Unique sequences: {stats['n_unique']}")
        print(f"  Scans with PSM: {stats['n_scans_with_psm']}")

        if stats['mod_distribution']:
            print(f"\n  Modification distribution:")
            for mod_name, count in stats['mod_distribution'].most_common():
                supported = any(s.lower() in mod_name.lower()
                                for s in SUPPORTED_MODS)
                tag = "" if supported else " ← UNSUPPORTED, will skip"
                print(f"    {mod_name}: {count} spectra{tag}")

        if stats['unsupported_mod_details']:
            print(f"\n  Unsupported mod breakdown:")
            for mod_name, count in sorted(stats['unsupported_mod_details'].items()):
                print(f"    {mod_name}: {count}")

        capped = min(stats['n_scans_with_psm'], args.max_spectra)
        print(f"\n  Expected output: {stats['n_scans_with_psm']} spectra "
              f"(capped to {capped} by --max_spectra)")
        print()

        all_sequences.update(stats['unique_sequences'])
        species_psms[species].update(psms)
        species_stats[species] = stats

    # ── Summary ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_expected = 0
    for species in sorted(species_psms.keys()):
        n = len(species_psms[species])
        capped = min(n, args.max_spectra)
        total_expected += capped
        print(f"  {species.replace('_', ' ')}: {n} scans "
              f"(capped to {capped})")
    print(f"  Total expected spectra: {total_expected}")
    print(f"  Total unique sequences across all species: {len(all_sequences)}")

    total_unsupported = sum(s.get('unsupported_mods', 0)
                            for s in species_stats.values())
    if total_unsupported > 0:
        print(f"  Skipped due to unsupported modifications: {total_unsupported} total")

    # ── Contamination audit ──────────────────────────────────────────────────

    overlap_pct = run_contamination_audit(all_sequences, args.training_msp_dir)

    if overlap_pct > 30:
        print(f"\nERROR: Sequence overlap {overlap_pct:.1f}% exceeds 30% threshold.")
        print("Aborting — do NOT proceed to evaluation.")
        sys.exit(1)

    # ── Dry run stops here ───────────────────────────────────────────────────

    if args.dry_run:
        print("\n*** DRY RUN complete. Remove --dry_run to write MGF files. ***")
        return

    # ── Full conversion: parse mzML and write MGF ────────────────────────────

    print("\n" + "=" * 60)
    print("FULL CONVERSION — parsing mzML and writing MGF")
    print("=" * 60)

    total_written = 0
    total_missing = 0

    for species in sorted(species_psms.keys()):
        psms = species_psms[species]
        if not psms:
            continue

        # Find the corresponding mzML file
        # The mzid filename is <base>_msgfplus.mzid.gz, mzML is <base>.mzML.gz
        mzml_path = None
        for mzid_path in mzid_files:
            sp = filename_to_species(mzid_path.name.replace('_msgfplus.mzid.gz', ''))
            if sp == species:
                base = mzid_path.name.replace('_msgfplus.mzid.gz', '')
                candidate = raw_dir / f"{base}.mzML.gz"
                if candidate.exists():
                    mzml_path = candidate
                    break

        if mzml_path is None:
            print(f"\nWARNING: No mzML file found for {species} — skipping")
            continue

        print(f"\n[{species}] Parsing {mzml_path.name} ...")
        scan_numbers = set(psms.keys())
        spectra = parse_mzml_spectra(str(mzml_path), scan_numbers)

        out_path = out_dir / f"{species}.mgf"
        print(f"  Writing {out_path.name} ...")
        written, missing = write_mgf(psms, spectra, str(out_path), args.max_spectra)
        total_written += written
        total_missing += missing
        print(f"  Written: {written} spectra, Missing from mzML: {missing}")

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"  Total spectra written: {total_written}")
    print(f"  Total scans missing from mzML: {total_missing}")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    for mgf in sorted(out_dir.glob('*.mgf')):
        size_mb = mgf.stat().st_size / (1024 * 1024)
        print(f"  {mgf.name} ({size_mb:.1f} MB)")
    print(f"\nNext step: sbatch scripts/submit_convert_hcd.sh (without --dry_run)")
    print(f"Then: eval_bacterial_hcd.py on login node")


if __name__ == '__main__':
    main()
