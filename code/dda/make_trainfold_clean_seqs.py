"""
make_trainfold_clean_seqs.py
============================
Recompute the clean training-sequence list RESTRICTED to the 45 train-fold species
(after the committed species split), minus the PXD010613 test sequences.

The earlier train_sequences_clean.txt was over the FULL 51-species manifest. T3 trains
only on the train fold (180 MGFs / 45 species), so the leakage-cleaned sequence set must
be recomputed from those MGFs. Bare-sequence convention is convert_mod(SEQ)[0] on BOTH
sides (P8) -- identical to the manifest and the earlier leakage check.

Inputs (read-only):
  - t3_split/train_fold_mgfs.txt  : the 180 train-fold MGF paths
  - data/bacteria_hcd/*.mgf       : PXD010613 test set
Output (NEW file, does NOT overwrite the existing clean list):
  - train_sequences_clean_trainfold.txt

HARD SANITY: train-fold seqs are a subset of the full manifest, so train-fold ∩ test
MUST be <= the full-manifest overlap (700). If it comes back higher, that's a bug --
the script flags it loudly.

Reuses scripts/verify_overlap.py: convert_mod, parse_mgf_seqs_only.

Usage:
  python scripts/make_trainfold_clean_seqs.py \\
    --train_fold_list /path/to/cluster_scratch/peptdeep_data/pxd010000_train_mgf/t3_split/train_fold_mgfs.txt \\
    --test_mgf_dir /path/to/peptdeep_project/data/bacteria_hcd
"""

import argparse
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_overlap import convert_mod, parse_mgf_seqs_only

FULL_MANIFEST_OVERLAP = 700  # full 51-species manifest ∩ PXD010613; train-fold must be <= this


def bare_seqs_from_mgfs(mgf_paths, tag):
    seqs = set()
    n_spectra = 0
    for i, p in enumerate(mgf_paths, 1):
        recs = parse_mgf_seqs_only(str(p))
        for r in recs:
            b = convert_mod(r.get('SEQ', ''))[0]
            if b:
                seqs.add(b)
        n_spectra += len(recs)
        if i % 30 == 0 or i == len(mgf_paths):
            print(f"  [{tag}] {i}/{len(mgf_paths)} files  spectra={n_spectra}  unique={len(seqs)}",
                  flush=True)
    return seqs, n_spectra


def main():
    ap = argparse.ArgumentParser(
        description='Recompute clean training seqs for the 45 train-fold species (minus PXD010613).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--train_fold_list', required=True,
                    help='t3_split/train_fold_mgfs.txt (180 train-fold MGF paths)')
    ap.add_argument('--test_mgf_dir', required=True, help='PXD010613 test MGFs (data/bacteria_hcd)')
    ap.add_argument('--out_file', default=None,
                    help='Output (default: <train_fold_list dir>/train_sequences_clean_trainfold.txt)')
    args = ap.parse_args()

    fold_list = Path(args.train_fold_list)
    if not fold_list.exists():
        sys.exit(f"ERROR: train_fold_list not found: {fold_list}")
    out_file = Path(args.out_file) if args.out_file \
        else fold_list.with_name('train_sequences_clean_trainfold.txt')

    # ── Resolve the 180 train-fold MGF paths ────────────────────────────────────
    mgf_paths = []
    missing = 0
    for line in fold_list.read_text().splitlines():
        p = line.strip()
        if not p:
            continue
        if not Path(p).exists():
            print(f"  WARNING: missing MGF {p}")
            missing += 1
            continue
        mgf_paths.append(p)
    print(f"train-fold MGFs: {len(mgf_paths)}  (missing: {missing})", flush=True)

    # ── Train-fold bare sequences ───────────────────────────────────────────────
    train_seqs, n_tr = bare_seqs_from_mgfs(mgf_paths, 'train-fold')
    print(f"train-fold unique bare seqs (BEFORE subtraction): {len(train_seqs)}  "
          f"(from {n_tr} spectra)", flush=True)

    # ── Test bare sequences (same convention) ───────────────────────────────────
    test_dir = Path(os.path.expanduser(args.test_mgf_dir))
    test_mgfs = sorted(test_dir.glob('*.mgf'))
    if not test_mgfs:
        sys.exit(f"ERROR: no test MGFs in {test_dir}")
    test_seqs, _ = bare_seqs_from_mgfs(test_mgfs, 'test')
    print(f"test unique bare seqs (PXD010613): {len(test_seqs)}", flush=True)

    # ── Subtract leakage ────────────────────────────────────────────────────────
    overlap = train_seqs & test_seqs
    clean = train_seqs - test_seqs

    print("\n" + "=" * 64)
    print("TRAIN-FOLD CLEAN SEQUENCE REPORT")
    print(f"  train-fold unique (before)  : {len(train_seqs)}")
    print(f"  overlap with test (removed) : {len(overlap)}")
    print(f"  clean (after)               : {len(clean)}")
    if len(overlap) > FULL_MANIFEST_OVERLAP:
        print(f"\n  *** WARNING: overlap {len(overlap)} > full-manifest overlap "
              f"{FULL_MANIFEST_OVERLAP}. ***")
        print("  Train-fold seqs are a SUBSET of the full manifest, so this is impossible")
        print("  unless there is a bug (convention mismatch / wrong inputs). Investigate.")
    else:
        print(f"  (OK: overlap {len(overlap)} <= full-manifest {FULL_MANIFEST_OVERLAP}, as expected)")
    print("=" * 64)

    # ── Write NEW file (do not overwrite existing clean list) ───────────────────
    if out_file.exists():
        bak = out_file.with_name(out_file.name + f".bak_{date.today():%Y%m%d}")
        if bak.exists():
            sys.exit(f"ERROR: backup {bak.name} already exists. Move/rename it first.")
        shutil.copy2(out_file, bak)
        print(f"backed up existing {out_file.name} -> {bak.name}")
    tmp = out_file.with_name(out_file.name + '.tmp')
    with open(tmp, 'w') as f:
        for s in sorted(clean):
            f.write(s + '\n')
    os.replace(str(tmp), str(out_file))
    print(f"\nWrote {len(clean)} clean train-fold sequences -> {out_file}")


if __name__ == '__main__':
    main()
