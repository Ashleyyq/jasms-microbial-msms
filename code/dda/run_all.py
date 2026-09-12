"""
ALL-IN-ONE: Baseline + Transfer Learning + Comparison
Usage:
  python scripts/run_all.py --max_spectra 500          # pilot (500/species)
  python scripts/run_all.py --max_spectra 5000         # medium scale
  python scripts/run_all.py --max_spectra 0            # full dataset (all spectra)
  python scripts/run_all.py --max_spectra 5000 --epochs 50  # more epochs
"""

import os, sys, json, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# Data parsing
# ============================================================

def parse_mgf_file(mgf_path, max_spectra=None):
    spectra = []
    scan = 0
    with open(mgf_path, 'r') as f:
        cur = {}; mzs = []; ints = []
        for line in f:
            line = line.strip()
            if line == 'BEGIN IONS':
                cur = {}; mzs = []; ints = []; scan += 1
            elif line == 'END IONS':
                if cur and mzs and 'SEQ' in cur:
                    cur['mz'] = np.array(mzs, dtype='float64')
                    cur['intensity'] = np.array(ints, dtype='float64')
                    cur['_scan'] = scan
                    cur['_file'] = Path(mgf_path).stem
                    spectra.append(cur)
                    if max_spectra and len(spectra) >= max_spectra:
                        break
            elif '=' in line:
                k, v = line.split('=', 1); cur[k] = v
            elif line:
                parts = line.split()
                if len(parts) >= 2:
                    try: mzs.append(float(parts[0])); ints.append(float(parts[1]))
                    except: pass
    print(f"  Parsed {len(spectra)} spectra from {Path(mgf_path).name}")
    return spectra


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


def angular_similarity(pred, exp):
    mask = (pred > 1e-6) | (exp > 1e-6)
    if mask.sum() < 2: return 0.0
    p, e = pred[mask], exp[mask]
    n1, n2 = np.linalg.norm(p), np.linalg.norm(e)
    if n1 == 0 or n2 == 0: return 0.0
    cos = np.clip(np.dot(p, e) / (n1 * n2), -1, 1)
    return float(1 - 2 * np.arccos(cos) / np.pi)


def pearson_corr(pred, exp):
    mask = (pred > 1e-6) | (exp > 1e-6)
    if mask.sum() < 3: return 0.0
    p, e = pred[mask], exp[mask]
    if np.std(p) == 0 or np.std(e) == 0: return 0.0
    return float(np.corrcoef(p, e)[0, 1])


def match_peaks_to_fragments(frag_mzs, exp_mz, exp_int, tolerance=0.5):
    """Vectorized matching of experimental peaks to theoretical fragment m/z.

    Args:
        frag_mzs: (n_frag, n_ion) theoretical fragment m/z values
        exp_mz: (n_peaks,) experimental m/z values
        exp_int: (n_peaks,) experimental intensities (should be pre-normalized)
        tolerance: m/z matching tolerance in Da

    Returns:
        (n_frag, n_ion) matched experimental intensities
    """
    n_frag, n_ion = frag_mzs.shape
    if len(exp_mz) == 0:
        return np.zeros((n_frag, n_ion))

    # Broadcasting: (n_frag, n_ion, 1) vs (1, 1, n_peaks) -> (n_frag, n_ion, n_peaks)
    diffs = np.abs(frag_mzs[:, :, np.newaxis] - exp_mz[np.newaxis, np.newaxis, :])
    within_tol = diffs < tolerance
    matched = np.where(within_tol, exp_int[np.newaxis, np.newaxis, :], 0.0)
    result = matched.max(axis=2)  # (n_frag, n_ion)
    result[frag_mzs <= 0] = 0.0
    return result


# ============================================================
# Load and prepare data
# ============================================================

def load_data(data_dir, max_spectra):
    print("=" * 60)
    print("[DATA] Parsing bacterial MGF files...")
    print("=" * 60)
    mgf_files = sorted(Path(data_dir).glob("*.mgf"))
    if not mgf_files:
        print(f"ERROR: No MGF files in {data_dir}"); sys.exit(1)
    print(f"Found {len(mgf_files)} MGF file(s)")

    all_spec = []
    for f in mgf_files:
        all_spec.extend(parse_mgf_file(str(f), max_spectra))
    print(f"Total: {len(all_spec)}")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for s in all_spec:
        seq, mods, sites = convert_mod(s.get('SEQ', ''))
        if not seq or not all(a in valid_aa for a in seq): continue
        if len(seq) < 7 or len(seq) > 30: continue
        if "Unknown" in mods: continue
        ch = int(s.get('CHARGE', '2+').replace('+','').replace('-',''))
        if ch < 1 or ch > 6: continue
        rows.append({
            'sequence': seq, 'mods': mods, 'mod_sites': sites,
            'charge': ch, 'nce': 30.0, 'instrument': 'Lumos',
            'exp_mz': s['mz'], 'exp_intensity': s['intensity'],
            'scan': s['_scan'], 'source': s['_file'],
        })
    print(f"Valid: {len(rows)}, Skipped: {len(all_spec)-len(rows)}")
    return rows


# ============================================================
# BASELINE evaluation (FIXED for dict return type)
# ============================================================

def evaluate_predictions(rows, precursor_df, frag_mz_df, frag_intensity_df):
    """
    Compare AlphaPeptDeep predictions vs experimental spectra.
    Uses the PREDICTED fragment m/z values to match experimental peaks.
    """
    results = []

    t_eval = datetime.now()
    for idx in range(len(rows)):
        row = rows[idx]
        start = int(precursor_df.iloc[idx]['frag_start_idx'])
        stop = int(precursor_df.iloc[idx]['frag_stop_idx'])

        exp_mz = row['exp_mz']
        exp_int = row['exp_intensity']
        if np.max(exp_int) > 0:
            exp_int = exp_int / np.max(exp_int)

        # Vectorized: match experimental peaks to all fragment m/z at once
        frag_mzs = frag_mz_df.iloc[start:stop].values  # (n_frag, n_ion)
        matched_exp = match_peaks_to_fragments(frag_mzs, exp_mz, exp_int)

        # Use b_z1 + y_z1 for evaluation (consistent with prior experiments)
        b_col = frag_mz_df.columns.get_loc('b_z1')
        y_col = frag_mz_df.columns.get_loc('y_z1')

        pred_b = frag_intensity_df.iloc[start:stop]['b_z1'].values
        pred_y = frag_intensity_df.iloc[start:stop]['y_z1'].values
        exp_b = matched_exp[:, b_col]
        exp_y = matched_exp[:, y_col]

        pred_vec = np.concatenate([pred_b, pred_y])
        exp_vec = np.concatenate([exp_b, exp_y])

        if np.max(pred_vec) > 0:
            pred_vec = pred_vec / np.max(pred_vec)

        ang = angular_similarity(pred_vec, exp_vec)
        pear = pearson_corr(pred_vec, exp_vec)

        results.append({
            'sequence': row['sequence'], 'charge': row['charge'],
            'length': len(row['sequence']), 'source': row['source'],
            'angular_similarity': ang, 'pearson_correlation': pear,
            'n_pred_nonzero': int(np.sum(pred_vec > 1e-6)),
            'n_exp_nonzero': int(np.sum(exp_vec > 1e-6)),
        })

        if (idx + 1) % 5000 == 0:
            elapsed = (datetime.now() - t_eval).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (len(rows) - idx - 1) / rate
            m = np.mean([r['angular_similarity'] for r in results])
            print(f"  {idx+1}/{len(rows)}  AS={m:.4f}  "
                  f"({rate:.0f} spec/s, ~{remaining:.0f}s remaining)")

    return pd.DataFrame(results)


def run_baseline(rows, output_dir):
    print("\n" + "=" * 60)
    print("[BASELINE] Pre-trained model on bacterial data")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    from peptdeep.pretrained_models import ModelManager
    model_mgr = ModelManager()
    model_mgr.load_installed_models()
    print("Model loaded!")

    # Build precursor table with index to restore order after predict_all
    precursor_input = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in rows])
    precursor_input['_orig_idx'] = range(len(rows))

    print(f"Predicting MS2 for {len(precursor_input)} peptides...")
    result = model_mgr.predict_all(precursor_input, predict_items=['ms2'])

    # Extract from dict
    precursor_df = result['precursor_df']
    frag_mz_df = result['fragment_mz_df']
    frag_intensity_df = result['fragment_intensity_df']

    # Restore original order (predict_all may reorder by peptide length)
    if '_orig_idx' in precursor_df.columns:
        precursor_df = precursor_df.sort_values('_orig_idx').reset_index(drop=True)

    print(f"  precursor_df: {precursor_df.shape}")
    print(f"  fragment_mz_df: {frag_mz_df.shape}")
    print(f"  fragment_intensity_df: {frag_intensity_df.shape}")
    print(f"  Ion types: {frag_intensity_df.columns.tolist()}")
    print()

    # Evaluate
    print("Evaluating predictions vs experimental spectra...")
    results_df = evaluate_predictions(rows, precursor_df, frag_mz_df, frag_intensity_df)

    ang = results_df['angular_similarity'].values
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'AlphaPeptDeep v1.4.1 (pre-trained, human)',
        'n_spectra': len(results_df),
        'angular_similarity': {
            'mean': float(np.mean(ang)), 'median': float(np.median(ang)),
            'std': float(np.std(ang)),
            'q25': float(np.percentile(ang, 25)), 'q75': float(np.percentile(ang, 75)),
        },
        'pearson_correlation': {
            'mean': float(results_df['pearson_correlation'].mean()),
            'median': float(results_df['pearson_correlation'].median()),
        },
    }

    results_df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)
    with open(os.path.join(output_dir, 'baseline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Spectra: {len(results_df)}")
    print(f"Avg predicted non-zero ions:    {results_df['n_pred_nonzero'].mean():.1f}")
    print(f"Avg experimental non-zero ions: {results_df['n_exp_nonzero'].mean():.1f}")
    print()
    print(f"Angular Similarity:  Mean={summary['angular_similarity']['mean']:.4f}  "
          f"Median={summary['angular_similarity']['median']:.4f}  "
          f"Std={summary['angular_similarity']['std']:.4f}")
    print(f"Pearson Correlation: Mean={summary['pearson_correlation']['mean']:.4f}")
    print()
    print("Per-species:")
    for sp, g in results_df.groupby('source'):
        print(f"  {sp:45s} AS={g['angular_similarity'].mean():.4f}  (n={len(g)})")
    print("=" * 60)

    return model_mgr, summary, results_df


# ============================================================
# TRANSFER LEARNING (vectorized target building)
# ============================================================

def run_transfer_learning(rows, model_mgr, model_dir, epochs=20, lr=0.0001, batch_size=1024):
    print("\n" + "=" * 60)
    print("[TRANSFER LEARNING] Fine-tuning on bacterial data")
    print(f"  Spectra: {len(rows)}, Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print("=" * 60)
    os.makedirs(model_dir, exist_ok=True)

    import torch

    # Step 1: Get predicted fragment structure (m/z positions for training targets)
    precursor_input = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in rows])
    precursor_input['_orig_idx'] = range(len(rows))

    print("Predicting fragment m/z structure...")
    t_pred = datetime.now()
    result = model_mgr.predict_all(precursor_input, predict_items=['ms2'])
    precursor_df = result['precursor_df']
    frag_mz_df = result['fragment_mz_df']
    frag_intensity_df = result['fragment_intensity_df']
    if '_orig_idx' in precursor_df.columns:
        precursor_df = precursor_df.sort_values('_orig_idx').reset_index(drop=True)
    print(f"  Done in {(datetime.now() - t_pred).total_seconds():.1f}s")

    # Step 2: Build training targets (vectorized peak matching)
    print("Building training targets from experimental spectra...")
    t_target = datetime.now()
    ion_cols = frag_mz_df.columns.tolist()
    target_values = np.zeros_like(frag_intensity_df.values)

    for idx in range(len(rows)):
        row = rows[idx]
        start = int(precursor_df.iloc[idx]['frag_start_idx'])
        stop = int(precursor_df.iloc[idx]['frag_stop_idx'])

        exp_mz = row['exp_mz']
        exp_int = row['exp_intensity']   # RAW: do NOT whole-spectrum base-peak normalize (route b)

        frag_mzs = frag_mz_df.iloc[start:stop].values
        target_values[start:stop, :] = match_peaks_to_fragments(frag_mzs, exp_mz, exp_int)
        # Align target scale to APD's convention: divide this peptide's MODELED-GRID slice by
        # its own max, reproducing peptdeep.model.ms2.normalize_fragment_intensities
        # (v1.4.1 ms2.py:851-853). Only the normalization denominator changes vs the old
        # whole-spectrum base-peak division; match tolerance/grid/direct train() are unchanged.
        _grid_max = target_values[start:stop, :].max()
        if _grid_max > 0:
            target_values[start:stop, :] /= _grid_max

        if (idx + 1) % 10000 == 0:
            elapsed = (datetime.now() - t_target).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (len(rows) - idx - 1) / rate
            print(f"  {idx+1}/{len(rows)} ({rate:.0f} spec/s, ~{remaining:.0f}s remaining)")

    target_df = pd.DataFrame(target_values, columns=ion_cols)
    print(f"  Target building done in {(datetime.now() - t_target).total_seconds():.1f}s")

    # Step 3: Fine-tune the MS2 model
    ms2_model = model_mgr.ms2_model
    trained = False

    try:
        print(f"\nTraining ms2_model for {epochs} epochs...")
        t_train = datetime.now()
        ms2_model.train(
            precursor_df=precursor_df,
            fragment_intensity_df=target_df,
            epoch=epochs, lr=lr, batch_size=batch_size,
        )
        trained = True
        print(f"  Training done in {(datetime.now() - t_train).total_seconds():.1f}s")
    except Exception as e:
        print(f"  ms2_model.train() failed: {e}")
        try:
            ms2_model.train(precursor_df, target_df, epoch=epochs, lr=lr, batch_size=batch_size)
            trained = True
        except Exception as e2:
            print(f"  Positional args also failed: {e2}")
            import traceback; traceback.print_exc()

    # Step 4: Save model
    if trained:
        model_path = os.path.join(model_dir, 'ms2_finetuned.pth')
        try:
            if hasattr(ms2_model, 'save'):
                ms2_model.save(model_path)
            elif hasattr(ms2_model, 'model'):
                torch.save(ms2_model.model.state_dict(), model_path)
            print(f"  Model saved to: {model_path}")
        except Exception as e:
            print(f"  Warning saving model: {e}")
    else:
        print("  Training failed. Baseline result is still valid.")

    return trained


# ============================================================
# Evaluate fine-tuned model
# ============================================================

def run_finetuned_eval(rows, model_mgr, baseline_summary, model_dir, output_dir):
    print("\n" + "=" * 60)
    print("[COMPARE] Fine-tuned vs Baseline")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    # Re-predict with (hopefully fine-tuned) model
    precursor_input = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in rows])
    precursor_input['_orig_idx'] = range(len(rows))

    # Try loading fine-tuned weights
    import torch
    mp = os.path.join(model_dir, 'ms2_finetuned.pth')
    if os.path.exists(mp):
        try:
            ms2 = model_mgr.ms2_model
            if hasattr(ms2, 'load'):
                ms2.load(mp)
            elif hasattr(ms2, 'model'):
                ms2.model.load_state_dict(torch.load(mp, map_location='cpu'), strict=False)
            print(f"Loaded fine-tuned weights from {mp}")
        except Exception as e:
            print(f"Warning loading weights: {e}")

    result = model_mgr.predict_all(precursor_input, predict_items=['ms2'])
    precursor_df = result['precursor_df']
    frag_mz_df = result['fragment_mz_df']
    frag_intensity_df = result['fragment_intensity_df']
    if '_orig_idx' in precursor_df.columns:
        precursor_df = precursor_df.sort_values('_orig_idx').reset_index(drop=True)

    results_df = evaluate_predictions(rows, precursor_df, frag_mz_df, frag_intensity_df)

    ft_mean = results_df['angular_similarity'].mean()
    ft_median = results_df['angular_similarity'].median()
    bl_mean = baseline_summary['angular_similarity']['mean']
    bl_median = baseline_summary['angular_similarity']['median']

    print()
    print("=" * 60)
    print(f"{'':30s} {'Baseline':>12s} {'Fine-tuned':>12s} {'Change':>12s}")
    print("-" * 66)
    print(f"{'Angular Similarity (mean)':30s} {bl_mean:>12.4f} {ft_mean:>12.4f} {ft_mean-bl_mean:>+12.4f}")
    print(f"{'Angular Similarity (median)':30s} {bl_median:>12.4f} {ft_median:>12.4f} {ft_median-bl_median:>+12.4f}")
    print()
    if ft_mean > bl_mean:
        print(f"  Improvement: +{(ft_mean-bl_mean)/max(bl_mean,0.001)*100:.1f}%")
    print("=" * 60)

    results_df.to_csv(os.path.join(output_dir, 'finetuned_results.csv'), index=False)
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump({'baseline_mean_AS': bl_mean, 'finetuned_mean_AS': ft_mean}, f, indent=2)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar(['Baseline\n(Human)', 'Fine-tuned\n(Bacterial)'], [bl_mean, ft_mean],
                   color=['#ff7f7f', '#7fbf7f'], edgecolor='black', width=0.5)
        ax[0].set_ylabel('Mean Angular Similarity'); ax[0].set_title('Baseline vs Fine-tuned')
        ax[0].set_ylim(0, max(bl_mean, ft_mean) * 1.4)
        ax[0].axhline(0.5, color='green', ls=':', alpha=0.5, label='Target (0.5)'); ax[0].legend()
        for i, v in enumerate([bl_mean, ft_mean]):
            ax[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold')
        ax[1].hist(results_df['angular_similarity'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax[1].axvline(ft_mean, color='red', ls='--', label=f'Mean: {ft_mean:.3f}'); ax[1].legend()
        ax[1].set_xlabel('Angular Similarity'); ax[1].set_title('Fine-tuned Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
        print(f"Plot saved: {os.path.join(output_dir, 'comparison.png')}")
    except Exception as e:
        print(f"Plot failed: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.expanduser('/path/to/peptdeep_project/data/bacteria'))
    parser.add_argument('--output_dir', default=os.path.expanduser('/path/to/peptdeep_project/results'))
    parser.add_argument('--model_dir', default=os.path.expanduser('/path/to/peptdeep_project/models/finetuned'))
    parser.add_argument('--max_spectra', type=int, default=500,
                        help='Max spectra per species (0 = load all)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--skip_transfer', action='store_true')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of data held out for testing (0 = no split, old behavior)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split reproducibility')
    args = parser.parse_args()

    t0 = datetime.now()
    print(f"Started: {t0.strftime('%H:%M:%S')}\n")

    all_rows = load_data(args.data_dir, args.max_spectra)

    # ================================================================
    # Train / Test Split
    # ================================================================
    if args.test_ratio > 0:
        np.random.seed(args.seed)
        indices = np.random.permutation(len(all_rows))
        n_test = max(1, int(len(all_rows) * args.test_ratio))
        test_rows = [all_rows[i] for i in indices[:n_test]]
        train_rows = [all_rows[i] for i in indices[n_test:]]

        # Count per-species distribution in each split
        train_species = {}
        for r in train_rows:
            train_species[r['source']] = train_species.get(r['source'], 0) + 1
        test_species = {}
        for r in test_rows:
            test_species[r['source']] = test_species.get(r['source'], 0) + 1

        print(f"\n{'='*60}")
        print(f"[SPLIT] Train/Test Split (seed={args.seed})")
        print(f"{'='*60}")
        print(f"  Train set: {len(train_rows)} spectra")
        print(f"  Test set:  {len(test_rows)} spectra")
        print(f"  Ratio:     {len(train_rows)/len(all_rows)*100:.0f}% / "
              f"{len(test_rows)/len(all_rows)*100:.0f}%")
        print(f"\n  Per-species distribution:")
        all_species = sorted(set(list(train_species.keys()) + list(test_species.keys())))
        for sp in all_species:
            tr_n = train_species.get(sp, 0)
            te_n = test_species.get(sp, 0)
            print(f"    {sp:45s}  train={tr_n:4d}  test={te_n:4d}")
        print(f"{'='*60}")

        # Save split info for reproducibility
        split_info = {
            'seed': args.seed, 'test_ratio': args.test_ratio,
            'n_total': len(all_rows), 'n_train': len(train_rows), 'n_test': len(test_rows),
            'train_species': train_species, 'test_species': test_species,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Split info saved to: {os.path.join(args.output_dir, 'split_info.json')}")
    else:
        train_rows = all_rows
        test_rows = all_rows
        print(f"\n[SPLIT] No split (test_ratio=0) — train and test on ALL {len(all_rows)} spectra")

    # ================================================================
    # Step 1: Baseline — evaluate pre-trained model on TEST set
    # ================================================================
    print(f"\n>>> Evaluating BASELINE on {'TEST set' if args.test_ratio > 0 else 'ALL data'} "
          f"({len(test_rows)} spectra)")
    model_mgr, bl_summary, bl_df = run_baseline(
        test_rows, os.path.join(args.output_dir, 'baseline'))

    if not args.skip_transfer:
        # ============================================================
        # Step 2: Fine-tune on TRAIN set only
        # ============================================================
        print(f"\n>>> Fine-tuning on {'TRAIN set' if args.test_ratio > 0 else 'ALL data'} "
              f"({len(train_rows)} spectra)")
        ok = run_transfer_learning(
            train_rows, model_mgr, args.model_dir,
            args.epochs, args.lr, args.batch_size)
        if ok:
            # ========================================================
            # Step 3: Evaluate fine-tuned model on TEST set
            # ========================================================
            print(f"\n>>> Evaluating FINE-TUNED model on "
                  f"{'TEST set' if args.test_ratio > 0 else 'ALL data'} "
                  f"({len(test_rows)} spectra)")
            run_finetuned_eval(test_rows, model_mgr, bl_summary,
                               args.model_dir, os.path.join(args.output_dir, 'comparison'))

    dt = (datetime.now() - t0).total_seconds() / 60
    print(f"\nTotal time: {dt:.1f} min")
    print(f"Results in: {args.output_dir}")
