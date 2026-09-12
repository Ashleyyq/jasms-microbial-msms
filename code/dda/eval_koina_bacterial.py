"""
Evaluate Koina models on the bacterial test set, using the IDENTICAL data loading
and train/test split as run_all.py.

Key design decisions:
  - Imports load_data(), parse_mgf_file(), convert_mod(), angular_similarity()
    directly from run_all.py to guarantee consistency.
  - Replicates run_all.py split exactly:
      np.random.seed(seed)
      indices   = np.random.permutation(len(all_rows))
      n_test    = max(1, int(len(all_rows) * test_ratio))
      test_rows = [all_rows[i] for i in indices[:n_test]]
  - Ground truth: theoretical b/y z=1 m/z computed from sequence + mod masses,
    then matched to experimental peaks within ±0.5 Da (same tolerance as run_all.py).
  - --verify mode: loads AlphaPeptDeep, compares theoretical m/z vs frag_mz_df
    on first 3 test spectra; warns if diff > 0.01 Da.

Usage (on cluster, inside peptdeep conda env):
  # Quick sanity-check run (pilot scale, same as default run_all.py)
  python scripts/eval_koina_bacterial.py \\
      --data_dir /path/to/peptdeep_project/data/bacteria \\
      --max_spectra 500 \\
      --models Prosit_2019_intensity Prosit_2020_intensity_HCD

  # Verify m/z consistency with AlphaPeptDeep (requires GPU/CPU with peptdeep installed)
  python scripts/eval_koina_bacterial.py \\
      --data_dir /path/to/peptdeep_project/data/bacteria \\
      --max_spectra 500 --verify

  # Full bacteria_full dataset (WARNING: large test set; Koina rate limits may apply)
  python scripts/eval_koina_bacterial.py \\
      --data_dir /path/to/peptdeep_project/data/bacteria_full \\
      --max_spectra 500 \\
      --models Prosit_2020_intensity_HCD Prosit_2025_intensity_40PTM

Output: results/{model_name}_bacterial_baseline.json  (per model)
        results/bacterial_koina_summary.json           (all models, for compare_all_baselines.py)
"""

import argparse
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Import from run_all.py to guarantee identical data loading ─────────────────
# run_all.py uses 'if __name__ == __main__': so importing it is safe.
from run_all import (
    parse_mgf_file,         # noqa: F401  (used indirectly via load_data)
    convert_mod,            # noqa: F401  (used indirectly via load_data)
    load_data,              # data loading + filtering (instrument='Lumos', nce=30)
    angular_similarity,     # AS = 1 - (2/pi)*arccos(cos_sim)
    pearson_corr,           # Pearson correlation (same formula as APD calc_ms2_similarity)
)

# ── Import Koina inference helpers from eval_koina_prosit.py ──────────────────
from eval_koina_prosit import (
    probe_model,
    call_koina,
    extract_z1_from_annotation,
    extract_z1_from_intensities_fallback,
    BATCH_SIZE,
)

# ── Theoretical fragment m/z ──────────────────────────────────────────────────

# Monoisotopic residue masses (Da) — standard proteomics convention
AA_MASS = {
    'A':  71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G':  57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P':  97.05276, 'S':  87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V':  99.06841,
}

# Modification mass deltas (same modifications handled in run_all.py convert_mod)
MOD_MASS = {
    'Oxidation@M':           15.99491,
    'Carbamidomethyl@C':     57.02146,
    'Acetyl@Protein_N-term': 42.01057,
}

PROTON    = 1.007276   # Da
H2O       = 18.01056   # Da
TOLERANCE = 0.5        # Da — identical to match_peaks_to_fragments() in run_all.py


def _parse_mods(mods_str, mod_sites_str):
    """
    Parse AlphaPeptDeep mod notation into {position: delta_mass}.
    Position 0 = N-terminus (e.g. Acetyl@Protein_N-term).
    Position k ∈ 1..n = residue k (1-indexed).
    """
    result = {}
    if not mods_str:
        return result
    mods  = [m for m in mods_str.split(';') if m]
    sites = [s for s in (mod_sites_str or '').split(';') if s]
    for mod, site in zip(mods, sites):
        delta = MOD_MASS.get(mod, 0.0)
        if delta == 0.0:
            continue
        try:
            pos = int(site)
            result[pos] = result.get(pos, 0.0) + delta
        except ValueError:
            pass
    return result


def compute_fragment_mz(sequence, mods_str, mod_sites_str):
    """
    Compute theoretical b_z1 and y_z1 ion m/z for a peptide (charge=1).

    Formulas (standard proteomics, same as AlphaPeptDeep/pyteomics):
      b_k (z=1) = N-term_extra + Σ residues[0..k-1] + PROTON
      y_k (z=1) = Σ residues[n-k..n-1] + H2O + PROTON

    Returns:
      b_z1_mz : ndarray (n-1,)  b_z1_mz[i] = m/z of b(i+1) ion
      y_z1_mz : ndarray (n-1,)  y_z1_mz[i] = m/z of y(i+1) ion
                                 y(1)=last residue → y_z1_mz[0]
                                 y(n-1)=all-but-first → y_z1_mz[-1]
    """
    n = len(sequence)
    mod_delta   = _parse_mods(mods_str, mod_sites_str)
    nterm_extra = mod_delta.get(0, 0.0)   # N-terminal modification (site=0)

    residue_masses = np.array([
        AA_MASS.get(aa, 0.0) + mod_delta.get(k + 1, 0.0)
        for k, aa in enumerate(sequence)
    ])

    # b-ions: cumulative sum from N-terminus, skip position n (full peptide)
    b_cumsum = np.cumsum(residue_masses)
    b_z1_mz  = b_cumsum[:-1] + nterm_extra + PROTON

    # y-ions: cumulative sum from C-terminus, skip position n (full peptide)
    y_cumsum = np.cumsum(residue_masses[::-1])
    y_z1_mz  = y_cumsum[:-1] + H2O + PROTON

    return b_z1_mz, y_z1_mz


def compute_gt_z1(row, b_z1_mz, y_z1_mz):
    """
    Match experimental peaks to theoretical b/y z=1 positions within ±TOLERANCE Da.
    Returns normalized concatenated array [gt_b_z1 | gt_y_z1].
    Matches the logic of run_all.py evaluate_predictions() using frag_mz_df.
    """
    n_ions  = len(b_z1_mz)
    exp_mz  = np.asarray(row['exp_mz'],        dtype=np.float64)
    exp_int = np.asarray(row['exp_intensity'],  dtype=np.float64)

    if len(exp_int) == 0 or np.max(exp_int) == 0:
        return np.zeros(n_ions * 2)
    exp_int_norm = exp_int / np.max(exp_int)

    gt_b = np.zeros(n_ions)
    gt_y = np.zeros(n_ions)

    for i in range(n_ions):
        diffs = np.abs(exp_mz - b_z1_mz[i])
        if diffs.min() <= TOLERANCE:
            gt_b[i] = exp_int_norm[np.argmin(diffs)]

        diffs = np.abs(exp_mz - y_z1_mz[i])
        if diffs.min() <= TOLERANCE:
            gt_y[i] = exp_int_norm[np.argmin(diffs)]

    return np.concatenate([gt_b, gt_y])


# ── Verify mode ───────────────────────────────────────────────────────────────

def verify_mz(test_rows, n_check=3):
    """
    Compare compute_fragment_mz() against AlphaPeptDeep's frag_mz_df on
    the first n_check test spectra. Prints b_z1 m/z[:3] and diffs.
    Warns if any diff > 0.01 Da.
    """
    print("\n=== VERIFY MODE: theoretical m/z vs AlphaPeptDeep frag_mz_df ===")
    try:
        from peptdeep.pretrained_models import ModelManager
    except ImportError:
        print("  ERROR: AlphaPeptDeep not importable — activate 'peptdeep' conda env first.")
        return

    rows_check = test_rows[:n_check]
    precursor_input = pd.DataFrame([{
        'sequence':  r['sequence'],
        'mods':      r['mods'],
        'mod_sites': r['mod_sites'],
        'charge':    r['charge'],
        'nce':       r['nce'],
        'instrument': r['instrument'],
    } for r in rows_check])
    precursor_input['_orig_idx'] = range(len(rows_check))

    print("  Loading AlphaPeptDeep pretrained model...")
    model_mgr = ModelManager()
    model_mgr.load_installed_models()
    result = model_mgr.predict_all(precursor_input, predict_items=['ms2'])
    precursor_df = result['precursor_df']
    frag_mz_df   = result['fragment_mz_df']
    if '_orig_idx' in precursor_df.columns:
        precursor_df = precursor_df.sort_values('_orig_idx').reset_index(drop=True)

    all_ok = True
    for idx in range(n_check):
        row   = rows_check[idx]
        start = int(precursor_df.iloc[idx]['frag_start_idx'])
        stop  = int(precursor_df.iloc[idx]['frag_stop_idx'])

        apd_b = frag_mz_df.iloc[start:stop]['b_z1'].values    # AlphaPeptDeep
        my_b, my_y = compute_fragment_mz(row['sequence'], row['mods'], row['mod_sites'])

        k = min(3, len(apd_b), len(my_b))
        diff = np.abs(apd_b[:k] - my_b[:k])
        max_diff = diff.max() if k > 0 else float('nan')

        print(f"\n  [{idx}] seq={row['sequence']}  mods='{row['mods']}'  charge={row['charge']}")
        print(f"       b_z1[:3]  AlphaPeptDeep : {np.round(apd_b[:k], 5).tolist()}")
        print(f"       b_z1[:3]  My formula    : {np.round(my_b[:k],  5).tolist()}")
        print(f"       Diff (Da)               : {np.round(diff, 6).tolist()}")
        if max_diff > 0.01:
            print(f"       WARNING: max diff {max_diff:.5f} Da > 0.01 Da — "
                  "check AA_MASS or MOD_MASS table!")
            all_ok = False
        else:
            print(f"       OK: max diff {max_diff:.6f} Da ≤ 0.01 Da")

    print("\n  " + ("All checks PASSED." if all_ok else "Some checks FAILED — fix before trusting GT."))
    print("=== END VERIFY ===\n")


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_model_bacterial(model_name, test_rows, use_annotation, payload_builder):
    """
    Evaluate one Koina model on test_rows.
    Returns (as_list, source_list) aligned by spectrum index.
    """
    n           = len(test_rows)
    seqs        = [r['sequence']  for r in test_rows]
    charges     = [r['charge']    for r in test_rows]
    nces        = [r['nce']       for r in test_rows]
    instruments = [r['instrument'] for r in test_rows]
    sources     = [r['source']    for r in test_rows]

    # Pre-compute all theoretical m/z (cheap; avoids redundant calls in the batch loop)
    frag_mzs = [
        compute_fragment_mz(r['sequence'], r['mods'], r['mod_sites'])
        for r in test_rows
    ]

    as_list     = []
    pcc_list    = []
    source_list = []

    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        payload = payload_builder(
            seqs[i:end], charges[i:end], nces[i:end], instruments[i:end])
        res, err = call_koina(model_name, payload)
        if err:
            print(f"    Batch {i}-{end} error: {err[:200]}")
            continue

        outputs = {o["name"]: o for o in res.get("outputs", [])}

        # Locate intensity array (shape: batch × ions_per_pred)
        int_data = None
        for name in ("intensities", "intensity"):
            if name in outputs:
                int_data = np.array(outputs[name]["data"], dtype=np.float32)
                break
        if int_data is None:
            for o in res.get("outputs", []):
                d = o.get("data")
                if d and len(d) % (end - i) == 0:
                    int_data = np.array(d, dtype=np.float32)
                    break
        if int_data is None:
            print(f"    Batch {i}-{end}: intensity output not found, skipping.")
            continue

        # Locate annotation array (optional)
        ann_data = None
        if use_annotation:
            for name in ("annotation", "annotations", "fragment_annotation"):
                if name in outputs:
                    ann_data = outputs[name].get("data")
                    if ann_data:
                        break

        batch_n       = end - i
        ions_per_pred = int_data.size // batch_n   # 174 for Prosit, 112 for APD_ms2_generic, etc.
        if int_data.size % batch_n != 0:
            print(f"    Batch {i}-{end}: output size {int_data.size} not divisible "
                  f"by {batch_n} peptides; skipping batch.")
            continue
        preds = int_data.reshape(batch_n, ions_per_pred)
        for j in range(batch_n):
            n_ions        = len(seqs[i + j]) - 1
            b_z1_mz, y_z1_mz = frag_mzs[i + j]

            # Predicted z=1 intensities from Koina
            if use_annotation and ann_data and len(ann_data) >= (j + 1) * ions_per_pred:
                ann_slice = ann_data[j * ions_per_pred:(j + 1) * ions_per_pred]
                pred_z1 = extract_z1_from_annotation(preds[j], ann_slice, n_ions)
            else:
                pred_z1 = extract_z1_from_intensities_fallback(preds[j], n_ions)

            if np.max(pred_z1) > 0:
                pred_z1 = pred_z1 / np.max(pred_z1)

            # Ground truth: theoretical m/z → match experimental peaks
            gt_z1 = compute_gt_z1(test_rows[i + j], b_z1_mz, y_z1_mz)
            if np.max(gt_z1) > 0:
                gt_z1 = gt_z1 / np.max(gt_z1)

            as_list.append(angular_similarity(pred_z1, gt_z1))
            pcc_list.append(pearson_corr(pred_z1, gt_z1))
            source_list.append(sources[i + j])

    return as_list, pcc_list, source_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate Koina models on bacterial test set.\n'
            'Uses IDENTICAL load_data() and split as run_all.py.\n'
            '--data_dir and --max_spectra MUST match the run_all.py experiment '
            'you want to compare against.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data_dir', required=True,
        help='Bacterial MGF directory (same value as run_all.py --data_dir)')
    parser.add_argument(
        '--max_spectra', type=int, default=500,
        help='Max spectra per MGF file (0=all; must match run_all.py --max_spectra; default: 500)')
    parser.add_argument(
        '--test_ratio', type=float, default=0.2,
        help='Test split fraction (must match run_all.py --test_ratio; default: 0.2)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for split (must match run_all.py --seed; default: 42)')
    parser.add_argument(
        '--models', nargs='+',
        default=[
            'Prosit_2019_intensity',
            'Prosit_2020_intensity_HCD',
            'Prosit_2025_intensity_40PTM',
            'AlphaPeptDeep_ms2_generic',
        ],
        help='Koina model names to evaluate')
    parser.add_argument(
        '--out_dir', default='results',
        help='Output directory for JSON result files')
    parser.add_argument(
        '--verify', action='store_true',
        help='Compare theoretical m/z vs AlphaPeptDeep frag_mz_df on first 3 test spectra '
             '(requires AlphaPeptDeep/peptdeep to be installed)')
    args = parser.parse_args()

    # ── Load data — identical to run_all.py ──────────────────────────────────
    max_sp   = args.max_spectra if args.max_spectra > 0 else None
    all_rows = load_data(args.data_dir, max_sp)

    # ── Exact same split as run_all.py ────────────────────────────────────────
    np.random.seed(args.seed)
    indices    = np.random.permutation(len(all_rows))
    n_test     = max(1, int(len(all_rows) * args.test_ratio))
    test_rows  = [all_rows[i] for i in indices[:n_test]]
    train_rows = [all_rows[i] for i in indices[n_test:]]    # not used here

    n_species = len(set(r['source'] for r in test_rows))
    print(f"\n[SPLIT] seed={args.seed}, test_ratio={args.test_ratio}")
    print(f"  Total: {len(all_rows)}  Train: {len(train_rows)}  Test: {len(test_rows)}")
    print(f"  Test set covers {n_species} species.\n")

    # ── Optional m/z verification (exits after check; run without --verify for model eval) ──
    if args.verify:
        verify_mz(test_rows, n_check=3)
        sys.exit(0)

    # ── Evaluate each model ───────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in args.models:
        print(f"\n--- Model: {model_name} ---")
        probe_data, use_annotation, payload_builder, err = probe_model(model_name)
        if probe_data is None:
            print(f"  Probe failed: {err}")
            print(f"  Skipping {model_name}.")
            continue

        as_list, pcc_list, source_list = run_model_bacterial(
            model_name, test_rows, use_annotation, payload_builder)

        if not as_list:
            print(f"  No results for {model_name}.")
            continue

        as_arr  = np.array(as_list)
        pcc_arr = np.array(pcc_list)

        # Per-species breakdown
        per_species = {}
        for sp in sorted(set(source_list)):
            mask   = np.array([s == sp for s in source_list])
            sp_as  = as_arr[mask]
            sp_pcc = pcc_arr[mask]
            per_species[sp] = {
                'mean':         float(np.mean(sp_as)),
                'median':       float(np.median(sp_as)),
                'mean_pearson': float(np.mean(sp_pcc)),
                'pcc90':        float(np.mean(sp_pcc >= 0.9)),
                'n':            int(mask.sum()),
            }

        out = {
            'model':                  model_name,
            'dataset':                'bacterial_MassIVE_MSV000079053',
            'data_dir':               str(args.data_dir),
            'max_spectra_per_file':   args.max_spectra,
            'split_seed':             args.seed,
            'split_test_ratio':       args.test_ratio,
            'n_spectra':              len(as_list),
            'n_species':              len(per_species),
            'metric':                 'angular_similarity_z1_only',
            'mean':                   float(np.mean(as_arr)),
            'median':                 float(np.median(as_arr)),
            'std':                    float(np.std(as_arr)),
            'mean_pearson':           float(np.mean(pcc_arr)),
            'median_pearson':         float(np.median(pcc_arr)),
            'pcc90':                  float(np.mean(pcc_arr >= 0.9)),
            'pcc75':                  float(np.mean(pcc_arr >= 0.75)),
            'per_species':            per_species,
        }

        out_path = out_dir / f"{model_name}_bacterial_baseline.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

        print(f"  mean_AS={out['mean']:.4f}  median_AS={out['median']:.4f}  "
              f"mean_PCC={out['mean_pearson']:.4f}  "
              f"PCC90={out['pcc90']*100:.1f}%  n={len(as_list)}")
        print(f"  Saved -> {out_path}")

        sorted_sp = sorted(per_species.items(), key=lambda x: x[1]['mean'], reverse=True)
        print(f"  Per-species (top 3 / bottom 3):")
        for sp, v in sorted_sp[:3]:
            print(f"    {sp:<42s}  mean={v['mean']:.4f}  n={v['n']}")
        if len(sorted_sp) > 6:
            print(f"    ...")
        for sp, v in sorted_sp[-min(3, len(sorted_sp)):]:
            print(f"    {sp:<42s}  mean={v['mean']:.4f}  n={v['n']}")

        all_results[model_name] = out

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo models produced results.")
        return

    summary_path = out_dir / 'bacterial_koina_summary.json'
    summary = {
        name: {
            'mean':         r['mean'],
            'median':       r['median'],
            'std':          r['std'],
            'mean_pearson': r.get('mean_pearson', float('nan')),
            'pcc90':        r.get('pcc90', float('nan')),
            'n_spectra':    r['n_spectra'],
        }
        for name, r in all_results.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("BACTERIAL KOINA EVALUATION SUMMARY")
    print(f"  data_dir={args.data_dir}  max_spectra={args.max_spectra}"
          f"  seed={args.seed}  test_ratio={args.test_ratio}")
    print(f"{'='*80}")
    hdr = f"{'Model':<45s}  {'MeanAS':>7s}  {'MedAS':>7s}  {'Std':>6s}  {'MeanPCC':>8s}  {'PCC90%':>7s}  {'N':>6s}"
    print(hdr)
    print('-' * len(hdr))
    for name, r in sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True):
        print(f"{name:<45s}  {r['mean']:>7.4f}  {r['median']:>7.4f}  "
              f"{r['std']:>6.4f}  {r.get('mean_pearson', float('nan')):>8.4f}  "
              f"{r.get('pcc90', float('nan'))*100:>6.1f}%  {r['n_spectra']:>6d}")
    print(f"\nFull per-species results -> {summary_path}")
    print("Compare with run_all.py baseline AS to validate GT consistency.")


if __name__ == '__main__':
    main()
