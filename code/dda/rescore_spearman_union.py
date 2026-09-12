"""
rescore_spearman_union.py
=========================
Re-score Spearman on PXD010613 with Aivett's UNION-of-top-N convention, for BOTH the
zero-shot baseline and the fine-tune E=6 model. Does NOT retrain. It re-runs the
(deterministic) prediction because per-ion predictions were never persisted; it also
recomputes SA/cosine/PCC/PCC90 and prints them next to the stored JSON values to PROVE
they are unchanged (only the Spearman scoring convention changes).

Union Spearman (Aivett): for each peptide, take obs top-N and pred top-N by intensity;
form the union; each side's rank column = 1..N for its own top-N (1 = most intense),
N+1 for union ions not in that side's top-N; Spearman between the two rank columns.
Also record union size per peptide (varies; N..2N).

Same protocol as the baseline/headline eval: instrument=QE, T1-identical per-species split
(seed+file_idx), tol 0.5Da, divide-by-max, b/y z1, n=3883.

Usage:
  python scripts/rescore_spearman_union.py \\
    --data_dir /path/to/peptdeep_project/data/bacteria_hcd \\
    --finetune_ckpt /path/to/peptdeep_project/models/t3_finetune_pxd010000_20260702/epoch_6/ms2_finetuned.pth \\
    --baseline_json results/local_zeroshot_baseline_qe_20260701/baseline_summary.json \\
    --finetune_json results/t3_finetune_eval_qe_20260702/baseline_summary.json \\
    --out_json results/rescore_union_qe_20260706.json --topn 7
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import angular_similarity, pearson_corr
from eval_bacterial_hcd import load_species_file            # QE loader + filters
from eval_koina_bacterial import compute_fragment_mz, compute_gt_z1
from eval_local_zeroshot_baseline import cosine, spearman_obs_topN, _load_finetuned_ms2, predict_species

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:
    _spearmanr = None


def spearman_union_topN(gt, pred, n):
    """Aivett union rule. Returns (rho, union_size)."""
    k = min(n, len(gt))
    obs_order = list(np.argsort(gt)[::-1][:k])     # highest observed first
    pred_order = list(np.argsort(pred)[::-1][:k])  # highest predicted first
    obs_rank = {ion: r + 1 for r, ion in enumerate(obs_order)}
    pred_rank = {ion: r + 1 for r, ion in enumerate(pred_order)}
    union = sorted(set(obs_order) | set(pred_order))
    if len(union) < 2:
        return np.nan, len(union)
    o = np.array([obs_rank.get(i, n + 1) for i in union], float)
    p = np.array([pred_rank.get(i, n + 1) for i in union], float)
    if np.all(o == o[0]) or np.all(p == p[0]):
        return np.nan, len(union)
    if _spearmanr is not None:
        rho = _spearmanr(o, p).correlation
        return (float(rho) if rho == rho else np.nan), len(union)
    import pandas as pd
    rr = np.corrcoef(pd.Series(o).rank(), pd.Series(p).rank())[0, 1]
    return (float(rr) if rr == rr else np.nan), len(union)


def eval_model(mm, data_dir, seed, test_ratio, max_spectra, topn):
    mgf_files = sorted(Path(os.path.expanduser(data_dir)).glob('*.mgf'))
    per_sp = {}
    allc = {'cos': [], 'sa': [], 'pcc': [], 'sp_obs': [], 'sp_uni': [], 'usize': []}
    for file_idx, mgf in enumerate(mgf_files):
        rows = load_species_file(mgf, max_spectra)
        if not rows:
            continue
        np.random.seed(seed + file_idx)                       # T1-identical split
        perm = np.random.permutation(len(rows))
        n_test = max(1, int(len(rows) * test_ratio))
        test_rows = [rows[i] for i in perm[:n_test]]
        pdf, fdf = predict_species(mm, test_rows)
        c = {'cos': [], 'sa': [], 'pcc': [], 'sp_obs': [], 'sp_uni': [], 'usize': []}
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
            pcc = pearson_corr(pred, gt)
            spu, us = spearman_union_topN(gt, pred, topn)
            c['cos'].append(cosine(pred, gt))
            c['sa'].append(float(angular_similarity(pred, gt)))
            c['pcc'].append(float(pcc) if pcc == pcc else np.nan)
            c['sp_obs'].append(spearman_obs_topN(gt, pred, topn))
            c['sp_uni'].append(spu)
            c['usize'].append(us)
        per_sp[mgf.stem] = _agg(c)
        for k in allc:
            allc[k].extend(c[k])
    return _agg(allc), per_sp


def _agg(c):
    a = {k: np.array(v, float) for k, v in c.items()}
    us = a['usize']
    return {
        'n': int(len(a['cos'])),
        'spearman_obs_top7': float(np.nanmean(a['sp_obs'])),
        'spearman_union_top7': float(np.nanmean(a['sp_uni'])),
        'n_spearman_union': int(np.sum(~np.isnan(a['sp_uni']))),
        'spectral_angle': float(np.nanmean(a['sa'])),
        'cosine': float(np.nanmean(a['cos'])),
        'pcc': float(np.nanmean(a['pcc'])),
        'pcc90': float(np.nanmean(a['pcc'] >= 0.9)),
        'union_size_mean': float(np.mean(us)) if len(us) else float('nan'),
        'union_size_min': int(np.min(us)) if len(us) else 0,
        'union_size_max': int(np.max(us)) if len(us) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--finetune_ckpt', required=True)
    ap.add_argument('--baseline_json', required=True)
    ap.add_argument('--finetune_json', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--max_spectra', type=int, default=5000)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topn', type=int, default=7)
    args = ap.parse_args()

    from peptdeep.pretrained_models import ModelManager

    print("=== ZERO-SHOT (re-predict) ===", flush=True)
    mm = ModelManager(); mm.load_installed_models()
    zs_over, zs_sp = eval_model(mm, args.data_dir, args.seed, args.test_ratio, args.max_spectra, args.topn)

    print("=== FINE-TUNE E=6 (re-predict) ===", flush=True)
    mm2 = ModelManager(); mm2.load_installed_models()
    _load_finetuned_ms2(mm2, args.finetune_ckpt)
    ft_over, ft_sp = eval_model(mm2, args.data_dir, args.seed, args.test_ratio, args.max_spectra, args.topn)

    zj = json.load(open(args.baseline_json))['overall']
    fj = json.load(open(args.finetune_json))['overall']

    def confirm(tag, rec, stored):
        print(f"\n[{tag}] confirm UNCHANGED metrics (recomputed vs stored JSON):")
        for k_new, k_old in [('spectral_angle', 'spectral_angle_mean'), ('cosine', 'cosine_mean'),
                             ('pcc', 'pcc_mean'), ('pcc90', 'pcc90'),
                             ('spearman_obs_top7', 'spearman_mean')]:
            d = rec[k_new] - stored[k_old]
            flag = 'OK' if abs(d) < 1e-6 else f'*** DIFF {d:+.2e}'
            print(f"  {k_new:<20} recomputed={rec[k_new]:.6f}  stored={stored[k_old]:.6f}  {flag}")

    confirm('zero-shot', zs_over, zj)
    confirm('fine-tune E6', ft_over, fj)

    print("\n" + "=" * 92)
    print("SPEARMAN: obs-top7 (old) vs UNION-top7 (new, Aivett)   [other 4 metrics unchanged]")
    print(f"{'scope':<26} {'model':<10} {'Sp_obs':>8} {'Sp_union':>9} {'usize(mean/min/max)':>22} {'n':>6}")
    def line(scope, model, rec):
        print(f"{scope:<26} {model:<10} {rec['spearman_obs_top7']:>8.4f} {rec['spearman_union_top7']:>9.4f} "
              f"{rec['union_size_mean']:>8.2f}/{rec['union_size_min']}/{rec['union_size_max']:<10} {rec['n']:>6}")
    line('OVERALL', 'zeroshot', zs_over)
    line('OVERALL', 'finetuneE6', ft_over)
    for sp in sorted(zs_sp):
        line(sp, 'zeroshot', zs_sp[sp])
        line(sp, 'finetuneE6', ft_sp[sp])
    print("=" * 92)

    out = {'convention': 'union of obs-top7 and pred-top7; non-top-N ion -> rank N+1; '
                          'Spearman between rank columns; union_size recorded per peptide',
           'topn': args.topn,
           'zero_shot': {'overall': zs_over, 'per_species': zs_sp},
           'fine_tune_E6': {'overall': ft_over, 'per_species': ft_sp}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == '__main__':
    main()
