"""
verify_overlap.py

Check whether the finetuned_v2 train set shares peptide sequences with
the evaluation test set.

Replicates run_all.py's load_data() and split logic exactly, but loads
only SEQ/CHARGE (skips peak data) to avoid OOM on login node.

Usage:
  python scripts/verify_overlap.py \
      --data_dir data/bacteria \
      --max_spectra_test 500 \
      --max_spectra_train 5000 \
      --seed 42 \
      --test_ratio 0.2
"""

import argparse
from pathlib import Path

import numpy as np

# ── Replicated from run_all.py (unchanged) ─────────────────────────────────────

def convert_mod(seq):
    clean = ""; mods = []; sites = []
    i = 0; pos = 0
    while i < len(seq):
        if seq[i].isupper():
            aa = seq[i]; clean += aa; pos += 1; i += 1
            if i < len(seq) and seq[i] == '+':
                j = i + 1
                while j < len(seq) and (seq[j].isdigit() or seq[j] == '.'): j += 1
                mass = float(seq[i+1:j])
                if abs(mass - 16.0) < 0.1 and aa == 'M':
                    mods.append("Oxidation@M"); sites.append(str(pos))
                elif abs(mass - 57.0) < 0.1 and aa == 'C':
                    mods.append("Carbamidomethyl@C"); sites.append(str(pos))
                else:
                    mods.append(f"Unknown@{aa}"); sites.append(str(pos))
                i = j
        else: i += 1
    return clean, ";".join(mods), ";".join(sites)


# ── Lightweight MGF reader (SEQ/CHARGE only, no peak arrays) ───────────────────

def parse_mgf_seqs_only(mgf_path, max_spectra=None):
    """
    Read MGF extracting only SEQ= and CHARGE= fields.
    Skips peak data entirely to stay memory-light on login node.
    """
    records = []
    cur = {}
    with open(mgf_path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line == 'BEGIN IONS':
                cur = {}
            elif line == 'END IONS':
                if 'SEQ' in cur:
                    records.append(cur)
                    if max_spectra and len(records) >= max_spectra:
                        break
                cur = {}
            elif '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                if k in ('SEQ', 'CHARGE'):
                    cur[k] = v.strip()
    return records


def load_seqs(data_dir, max_spectra):
    """
    Equivalent of run_all.py's load_data() but returns only
    list of {'sequence': str, 'charge': int} dicts.
    Filtering rules are identical to load_data().
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    mgf_files = sorted(Path(data_dir).glob("*.mgf"))
    print(f"  Found {len(mgf_files)} MGF files")

    all_spec = []
    for f in mgf_files:
        all_spec.extend(parse_mgf_seqs_only(str(f), max_spectra))
    print(f"  Raw spectra: {len(all_spec)}")

    rows = []
    for s in all_spec:
        seq, mods, _ = convert_mod(s.get('SEQ', ''))
        if not seq or not all(a in valid_aa for a in seq):
            continue
        if len(seq) < 7 or len(seq) > 30:
            continue
        if "Unknown" in mods:
            continue
        ch_str = s.get('CHARGE', '2+').replace('+', '').replace('-', '')
        try:
            ch = int(ch_str)
        except ValueError:
            continue
        if ch < 1 or ch > 6:
            continue
        rows.append({'sequence': seq, 'charge': ch})

    print(f"  Valid rows: {len(rows)}")
    return rows


def do_split(rows, seed, test_ratio):
    """Exact replica of run_all.py split logic."""
    np.random.seed(seed)
    indices = np.random.permutation(len(rows))
    n_test  = max(1, int(len(rows) * test_ratio))
    test_rows  = [rows[i] for i in indices[:n_test]]
    train_rows = [rows[i] for i in indices[n_test:]]
    return train_rows, test_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',          default='data/bacteria')
    ap.add_argument('--max_spectra_test',  type=int, default=500)
    ap.add_argument('--max_spectra_train', type=int, default=5000)
    ap.add_argument('--seed',              type=int, default=42)
    ap.add_argument('--test_ratio',        type=float, default=0.2)
    args = ap.parse_args()

    # ── Load test set (500/species, same as eval) ──────────────────────────────
    print(f"\n[1] Loading TEST data  (max_spectra={args.max_spectra_test})")
    rows_test_load = load_seqs(args.data_dir, args.max_spectra_test)
    _, test_rows = do_split(rows_test_load, args.seed, args.test_ratio)
    test_seqs = {r['sequence'] for r in test_rows}
    print(f"  Test set: {len(test_rows)} rows, {len(test_seqs)} unique sequences")

    # ── Load train set (5000/species, finetuned_v2) ────────────────────────────
    print(f"\n[2] Loading TRAIN data (max_spectra={args.max_spectra_train})")
    rows_train_load = load_seqs(args.data_dir, args.max_spectra_train)
    train_rows, _ = do_split(rows_train_load, args.seed, args.test_ratio)
    train_seqs = {r['sequence'] for r in train_rows}
    print(f"  Train set: {len(train_rows)} rows, {len(train_seqs)} unique sequences")

    # ── Overlap analysis ───────────────────────────────────────────────────────
    overlap = test_seqs & train_seqs
    pct     = len(overlap) / len(test_seqs) * 100 if test_seqs else 0.0

    print(f"\n{'='*60}")
    print(f"OVERLAP ANALYSIS")
    print(f"{'='*60}")
    print(f"  Train unique sequences : {len(train_seqs)}")
    print(f"  Test  unique sequences : {len(test_seqs)}")
    print(f"  Overlap count          : {len(overlap)}")
    print(f"  Overlap / test         : {pct:.2f}%")

    if overlap:
        print(f"\n  First {min(10, len(overlap))} overlapping sequences:")
        for seq in sorted(overlap)[:10]:
            print(f"    {seq}")
    else:
        print("\n  No overlap — train and test sets are sequence-disjoint.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
