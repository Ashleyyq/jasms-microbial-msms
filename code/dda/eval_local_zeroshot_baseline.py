"""
eval_local_zeroshot_baseline.py
===============================
THE local zero-shot baseline on PXD010613 (data/bacteria_hcd) that the T3 fine-tune
will be compared against. Local APD pretrained checkpoint, ZERO-SHOT.

CRITICAL CONFIG (repeat-bug guard P11/D14): instrument = QE (NOT Lumos). We reuse T1's
loader eval_bacterial_hcd.load_species_file, which already pins instrument='QE', NCE=30,
and the len 7-30 / charge 1-6 / no-Unknown filters, and we replicate T1's PER-SPECIES
deterministic split (seed+file_idx, test_ratio 0.2, max_spectra 5000) so this baseline is
directly comparable to the Koina QE reference (AS~0.71) and to the future fine-tune eval.

GT = compute_gt_z1 (obs matched to theoretical b/y z1 m/z, tol 0.5 Da, divide-by-base-peak).
Prediction = local APD, b/y z1, y reversed to ascending, divide-by-max. Same ion set/GT as
the T1 eval. (Target-normalization fix is under review but is TRAINING-only; it does NOT
affect this zero-shot eval, and eval metrics are scale-invariant anyway.)

Metrics (D5), per-species and overall, with n:
  - Spearman, observed-defined top-N (N=7): PROVISIONAL convention (pending Aivett). Isolated
    in one function so ONLY this can be re-run if the convention changes.
  - spectral_angle (= angular_similarity), cosine, PCC + PCC90 : internal cross-checks.

Usage:
  python scripts/eval_local_zeroshot_baseline.py \\
    --data_dir /path/to/peptdeep_project/data/bacteria_hcd \\
    --out_dir results/local_zeroshot_baseline_qe_<date> \\
    --max_spectra 5000 --test_ratio 0.2 --seed 42 --topn 7
"""

import argparse
import gc
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import angular_similarity, pearson_corr
from eval_bacterial_hcd import load_species_file            # pins instrument='QE'
from eval_koina_bacterial import compute_fragment_mz, compute_gt_z1

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:
    _spearmanr = None


def _load_finetuned_ms2(mm, path):
    """Load fine-tuned ms2 weights ON TOP of the installed v3 model (for the finetune eval)."""
    import torch
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: finetuned_model not found: {p}")
    ms2 = mm.ms2_model
    try:
        ms2.load(str(p))
        print(f"Loaded fine-tuned ms2 weights via ms2.load(): {p}", flush=True)
    except Exception as e:
        state = torch.load(str(p), map_location='cpu')
        ms2.model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned ms2 via state_dict(strict=False): {p} (ms2.load failed: {e})", flush=True)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def spearman_obs_topN(gt, pred, n):
    """PROVISIONAL convention: rank-correlate obs vs pred over the top-N OBSERVED ions."""
    order = np.argsort(gt)[::-1][:n]        # top-N by observed intensity
    g, p = gt[order], pred[order]
    if len(g) < 2 or np.all(g == g[0]) or np.all(p == p[0]):
        return np.nan
    if _spearmanr is not None:
        rho = _spearmanr(g, p).correlation
        return float(rho) if rho == rho else np.nan   # nan-safe
    # fallback: Pearson on ranks
    gr = pd.Series(g).rank().values
    pr = pd.Series(p).rank().values
    if np.std(gr) == 0 or np.std(pr) == 0:
        return np.nan
    return float(np.corrcoef(gr, pr)[0, 1])


def predict_species(mm, test_rows):
    """Local APD prediction for one species' test rows; returns fragment dfs."""
    precursor_input = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in test_rows])
    precursor_input['_orig_idx'] = range(len(test_rows))
    result = mm.predict_all(precursor_input, predict_items=['ms2'])
    pdf = result['precursor_df']
    if '_orig_idx' in pdf.columns:
        pdf = pdf.sort_values('_orig_idx').reset_index(drop=True)
    return pdf, result['fragment_intensity_df']


def main():
    ap = argparse.ArgumentParser(
        description='Local zero-shot baseline on PXD010613 (QE), all metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--max_spectra', type=int, default=5000)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topn', type=int, default=7, help='N for observed-defined top-N Spearman')
    ap.add_argument('--finetuned_model', default=None,
                    help='Path to fine-tuned ms2 .pth; if given, loaded on top of the v3 model')
    ap.add_argument('--label', default='zeroshot', help='Output label, e.g. finetune_epoch10')
    args = ap.parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir))
    out_dir = Path(args.out_dir) if args.out_dir \
        else Path(f"results/local_zeroshot_baseline_qe_{date.today():%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mgf_files = sorted(data_dir.glob('*.mgf'))
    if not mgf_files:
        sys.exit(f"ERROR: no MGFs in {data_dir}")

    from peptdeep.pretrained_models import ModelManager
    mm = ModelManager()
    mm.load_installed_models()
    if args.finetuned_model:
        _load_finetuned_ms2(mm, args.finetuned_model)

    per_species = {}
    all_recs = []   # (species, cos, sa, pcc, spearman)

    for file_idx, mgf in enumerate(mgf_files):
        species = mgf.stem
        rows = load_species_file(mgf, args.max_spectra)     # instrument='QE'
        if not rows:
            print(f"[{species}] 0 valid rows, skip", flush=True)
            continue
        # replicate T1 per-species split
        np.random.seed(args.seed + file_idx)
        perm = np.random.permutation(len(rows))
        n_test = max(1, int(len(rows) * args.test_ratio))
        test_rows = [rows[i] for i in perm[:n_test]]
        print(f"[{species}] rows={len(rows)} test={len(test_rows)}  predicting...", flush=True)

        pdf, fdf = predict_species(mm, test_rows)
        if 'b_z1' not in fdf.columns:
            sys.exit(f"ERROR: 'b_z1' missing in prediction cols: {list(fdf.columns)}")

        recs = []
        for idx, r in enumerate(test_rows):
            start = int(pdf.iloc[idx]['frag_start_idx'])
            stop = int(pdf.iloc[idx]['frag_stop_idx'])
            n_ions = stop - start
            if n_ions <= 0:
                continue
            b_mz, y_mz = compute_fragment_mz(r['sequence'], r['mods'], r['mod_sites'])
            gt = compute_gt_z1(r, b_mz[:n_ions], y_mz[:n_ions])
            if len(gt) != 2 * n_ions or np.max(gt) == 0:
                continue
            pb = fdf.iloc[start:stop]['b_z1'].values[:n_ions].astype(float)
            py = fdf.iloc[start:stop]['y_z1'].values[:n_ions].astype(float)
            pred = np.concatenate([pb, py[::-1]])
            if np.max(pred) <= 0:
                continue
            pred = pred / np.max(pred)

            cos = cosine(pred, gt)
            sa = float(angular_similarity(pred, gt))
            pcc = pearson_corr(pred, gt)
            sp = spearman_obs_topN(gt, pred, args.topn)
            recs.append((species, cos, sa, float(pcc) if pcc == pcc else np.nan, sp))

        all_recs.extend(recs)
        per_species[species] = summarize(recs, args.topn)
        s = per_species[species]
        print(f"  n={s['n']}  spearman_top{args.topn}={s['spearman_mean']:.4f}  "
              f"SA={s['spectral_angle_mean']:.4f}  cos={s['cosine_mean']:.4f}  "
              f"PCC={s['pcc_mean']:.4f}  PCC90={s['pcc90']*100:.1f}%", flush=True)
        del rows, test_rows, pdf, fdf
        gc.collect()

    overall = summarize(all_recs, args.topn)

    out = {
        'dataset': 'PXD010613 (data/bacteria_hcd)',
        'model': f'AlphaPeptDeep_v1.4.1_v3_{args.label}',
        'finetuned_model': args.finetuned_model,
        'config': {'instrument': 'QE', 'nce': 30.0, 'tolerance_Da': 0.5,
                   'norm': 'divide-by-max', 'len': '7-30', 'charge': '1-6',
                   'max_spectra': args.max_spectra, 'test_ratio': args.test_ratio,
                   'seed': args.seed, 'split': 'per-species seed+file_idx (T1-identical)'},
        'headline_metric': f'spearman_obs_top{args.topn}_PROVISIONAL',
        'provisional_note': ('Spearman convention = observed-defined top-N (N=%d); '
                             'PROVISIONAL pending Aivett confirmation. If the convention '
                             'changes, only spearman_obs_topN needs re-running.' % args.topn),
        'overall': overall,
        'per_species': per_species,
    }
    with open(out_dir / 'baseline_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print("LOCAL ZERO-SHOT BASELINE (QE)  --  headline = Spearman obs-top%d [PROVISIONAL]" % args.topn)
    print("=" * 78)
    hdr = f"{'species':<26} {'n':>6} {'Spearman':>9} {'SpecAngle':>10} {'cosine':>8} {'PCC':>7} {'PCC90':>7}"
    print(hdr)
    for sp in sorted(per_species):
        s = per_species[sp]
        print(f"{sp:<26} {s['n']:>6} {s['spearman_mean']:>9.4f} {s['spectral_angle_mean']:>10.4f} "
              f"{s['cosine_mean']:>8.4f} {s['pcc_mean']:>7.4f} {s['pcc90']*100:>6.1f}%")
    print("-" * 78)
    print(f"{'OVERALL':<26} {overall['n']:>6} {overall['spearman_mean']:>9.4f} "
          f"{overall['spectral_angle_mean']:>10.4f} {overall['cosine_mean']:>8.4f} "
          f"{overall['pcc_mean']:>7.4f} {overall['pcc90']*100:>6.1f}%")
    print("=" * 78)
    print(f"n_spearman (non-nan) overall: {overall['n_spearman']}")
    print(f"\nWrote {out_dir / 'baseline_summary.json'}")


def summarize(recs, topn):
    if not recs:
        return {'n': 0, 'spearman_mean': float('nan'), 'spectral_angle_mean': float('nan'),
                'cosine_mean': float('nan'), 'pcc_mean': float('nan'), 'pcc90': float('nan'),
                'n_spearman': 0}
    cos = np.array([r[1] for r in recs], float)
    sa = np.array([r[2] for r in recs], float)
    pcc = np.array([r[3] for r in recs], float)
    sp = np.array([r[4] for r in recs], float)
    return {
        'n': len(recs),
        'spearman_mean': float(np.nanmean(sp)),
        'n_spearman': int(np.sum(~np.isnan(sp))),
        'spectral_angle_mean': float(np.nanmean(sa)),
        'cosine_mean': float(np.nanmean(cos)),
        'pcc_mean': float(np.nanmean(pcc)),
        'pcc90': float(np.nanmean(pcc >= 0.9)),
    }


if __name__ == '__main__':
    main()
