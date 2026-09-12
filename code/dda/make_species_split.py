"""
make_species_split.py
======================
Species-level held-out split of PXD010000 for the T3 fine-tune.

GROUPING IS AUTHORITATIVE (metadata, not filenames -- P1): each run's organism is the
MS-GF+ search-database id (<SearchDatabase location="...ID_#####_XXXX.fasta">) from its
mzid. Confirmed 51 distinct ids over 235 runs. No same-organism run can straddle the
split because fold assignment is keyed on this id, never on file index (P6).

The held-out fold is an INTERNAL VALIDATION fold (epoch/LR selection, in-domain
generalization). The final test set is PXD010613 (separate, species-disjoint).

Decisions (pinned with user 2026-06-29):
  - hold out --n_holdout=6 organisms, random with --seed=42,
  - chosen ONLY from organisms with >= --min_runs (=5) runs AND >= --min_spectra spectra
    (the candidate pool), so validation species are data-rich.

TWO MODES:
  REPORT (default): print the per-organism table (n_runs, n_spectra, example run) and the
    candidate pool. Writes NOTHING. Review this, then pick --min_spectra.
  COMMIT (--commit): randomly pick the held-out organisms from the pool and write
    train_fold_mgfs.txt + val_fold_mgfs.txt + holdout_species.txt to --out_dir.
    (Backs up any existing list to .bak_<date> first -- no git.)

Grouping source: pass --run_dbid_map (the "<dbid> <base>" file you already generated) to
reuse the verified mapping; otherwise the script extracts dbid from each mzid in --raw_dir.

Usage -- report first:
  python scripts/make_species_split.py \\
    --mgf_dir /path/to/cluster_scratch/peptdeep_data/pxd010000_train_mgf \\
    --run_dbid_map ~/run_to_dbid.txt --min_runs 5
Usage -- commit after choosing --min_spectra:
  python scripts/make_species_split.py \\
    --mgf_dir /path/to/cluster_scratch/peptdeep_data/pxd010000_train_mgf \\
    --run_dbid_map ~/run_to_dbid.txt --min_runs 5 --min_spectra <N> \\
    --n_holdout 6 --seed 42 --commit --out_dir /path/to/cluster_scratch/peptdeep_data/pxd010000_train_mgf/t3_split
"""

import argparse
import gzip
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

MZID_SUFFIX = '_msgfplus.mzid.gz'
DBID_RE = re.compile(r'ID_[0-9]+_[A-Z0-9]+')


def extract_dbid_from_mzid(mzid_path):
    """First SearchDatabase FASTA id (ID_#####_XXXX) in the mzid; None if absent."""
    with gzip.open(mzid_path, 'rt', errors='replace') as fh:
        for line in fh:
            if '.fasta' in line and 'SearchDatabase' in line:
                m = DBID_RE.search(line)
                if m:
                    return m.group(0)
    return None


def count_spectra(mgf_path):
    n = 0
    with open(mgf_path, errors='replace') as fh:
        for line in fh:
            if line.startswith('BEGIN IONS'):
                n += 1
    return n


def load_dbid_map(path):
    """Read a '<dbid> <base>' per-line file into {base: dbid}."""
    m = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith('ID_'):
                m[parts[1]] = parts[0]
    return m


def write_list(path: Path, items):
    if path.exists():
        bak = path.with_name(path.name + f".bak_{date.today():%Y%m%d}")
        if bak.exists():
            sys.exit(f"ERROR: backup {bak.name} already exists. Move/rename it first.")
        shutil.copy2(path, bak)
        print(f"  backed up {path.name} -> {bak.name}")
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        for it in items:
            f.write(it + '\n')
    os.replace(str(tmp), str(path))


def main():
    ap = argparse.ArgumentParser(
        description='Species-level held-out split of PXD010000 (report, then commit).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--mgf_dir', required=True, help='Dir with the 235 built MGFs')
    ap.add_argument('--raw_dir', default=None,
                    help='Dir with mzid (to extract dbid); not needed if --run_dbid_map given')
    ap.add_argument('--run_dbid_map', default=None,
                    help='Pre-built "<dbid> <base>" file to reuse the verified grouping')
    ap.add_argument('--min_runs', type=int, default=5,
                    help='Held-out candidate pool: organisms with >= this many runs')
    ap.add_argument('--min_spectra', type=int, default=0,
                    help='Held-out candidate pool: organisms with >= this many spectra')
    ap.add_argument('--n_holdout', type=int, default=6, help='How many organisms to hold out')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed for the random pick')
    ap.add_argument('--commit', action='store_true',
                    help='Write the split (default off = report only)')
    ap.add_argument('--out_dir', default=None,
                    help='Where to write the fold lists (default: <mgf_dir>/t3_split)')
    args = ap.parse_args()

    mgf_dir = Path(args.mgf_dir)

    # ── Resolve base -> dbid (organism) grouping ────────────────────────────────
    base_to_dbid = {}
    if args.run_dbid_map and Path(os.path.expanduser(args.run_dbid_map)).exists():
        base_to_dbid = load_dbid_map(os.path.expanduser(args.run_dbid_map))
        print(f"Loaded grouping from {args.run_dbid_map}: {len(base_to_dbid)} runs")
    elif args.raw_dir:
        raw = Path(args.raw_dir)
        mzids = sorted(raw.glob('*' + MZID_SUFFIX))
        if not mzids:
            sys.exit(f"ERROR: no mzid in {raw}")
        print(f"Extracting dbid from {len(mzids)} mzid (this reads mzid headers)...", flush=True)
        for mz in mzids:
            base = mz.name[:-len(MZID_SUFFIX)]
            dbid = extract_dbid_from_mzid(str(mz))
            if dbid is None:
                print(f"  WARNING: no DB id in {mz.name}")
                continue
            base_to_dbid[base] = dbid
    else:
        sys.exit("ERROR: provide --run_dbid_map or --raw_dir for the organism grouping.")

    # ── Count spectra per MGF, aggregate per organism ───────────────────────────
    org = defaultdict(lambda: {'runs': [], 'spectra': 0})
    n_missing = 0
    for base, dbid in sorted(base_to_dbid.items()):
        mgf = mgf_dir / f"{base}.mgf"
        if not mgf.exists():
            print(f"  WARNING: no MGF for {base}")
            n_missing += 1
            continue
        ns = count_spectra(str(mgf))
        org[dbid]['runs'].append((base, str(mgf), ns))
        org[dbid]['spectra'] += ns

    # ── Report table ────────────────────────────────────────────────────────────
    rows = sorted(((d, len(v['runs']), v['spectra'], v['runs'][0][0])
                   for d, v in org.items()), key=lambda r: -r[2])
    print(f"\n{'DB id':<22} {'runs':>5} {'spectra':>10}  example_run")
    print("-" * 100)
    tot_runs = tot_spec = 0
    for dbid, nr, ns, ex in rows:
        tot_runs += nr
        tot_spec += ns
        print(f"{dbid:<22} {nr:>5} {ns:>10}  {ex[:55]}")
    print("-" * 100)
    print(f"organisms: {len(rows)}   runs: {tot_runs}   spectra: {tot_spec}   "
          f"missing MGF: {n_missing}")

    pool = sorted(d for d, nr, ns, ex in rows if nr >= args.min_runs and ns >= args.min_spectra)
    print(f"\nCandidate held-out pool (runs>={args.min_runs} AND spectra>={args.min_spectra}): "
          f"{len(pool)} organisms")
    for d in pool:
        print(f"  {d}  runs={len(org[d]['runs'])}  spectra={org[d]['spectra']}  "
              f"ex={org[d]['runs'][0][0][:50]}")

    if not args.commit:
        print("\nREPORT MODE -- nothing written.")
        print("Review the spectra column, pick --min_spectra, then re-run with --commit.")
        return

    # ── Commit the split ────────────────────────────────────────────────────────
    if len(pool) < args.n_holdout:
        sys.exit(f"ERROR: candidate pool ({len(pool)}) < n_holdout ({args.n_holdout}). "
                 f"Lower --min_spectra/--min_runs or --n_holdout.")
    random.seed(args.seed)
    val_ids = sorted(random.sample(pool, args.n_holdout))

    val_mgfs, train_mgfs = [], []
    for dbid, v in org.items():
        for base, mgf, ns in v['runs']:
            (val_mgfs if dbid in val_ids else train_mgfs).append(mgf)

    out_dir = Path(args.out_dir) if args.out_dir else mgf_dir / 't3_split'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_list(out_dir / 'train_fold_mgfs.txt', sorted(train_mgfs))
    write_list(out_dir / 'val_fold_mgfs.txt', sorted(val_mgfs))
    write_list(out_dir / 'holdout_species.txt',
               [f"{vid}\truns={len(org[vid]['runs'])}\tspectra={org[vid]['spectra']}"
                f"\tex={org[vid]['runs'][0][0]}" for vid in val_ids])

    print("\n" + "=" * 64)
    print(f"COMMIT: held out {len(val_ids)} organisms (seed {args.seed}) -> {out_dir}")
    print(f"  train fold: {len(train_mgfs)} MGFs   ({len(org)-len(val_ids)} organisms)")
    print(f"  val   fold: {len(val_mgfs)} MGFs   ({len(val_ids)} organisms)")
    for vid in val_ids:
        print(f"  HOLDOUT {vid}  runs={len(org[vid]['runs'])}  spectra={org[vid]['spectra']}  "
              f"ex={org[vid]['runs'][0][0][:50]}")
    print("=" * 64)
    print("\nNOTE: the clean training-sequence list must still be re-derived from the")
    print("train_fold MGFs (train-fold seqs minus PXD010613) before T3 uses it.")


if __name__ == '__main__':
    main()
