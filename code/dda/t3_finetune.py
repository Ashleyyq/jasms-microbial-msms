"""
t3_finetune.py
==============
T3 fine-tune of AlphaPeptDeep (v1.4.1 + v3 pretrained) on the PXD010000 train-fold, with
val_fold-based epoch selection. Compares later against the LOCAL zero-shot baseline only.

Locked decisions honored:
- Train data = train-fold MGFs (train_fold_mgfs.txt), filtered (len7-30/AA/charge1-6/no-Unknown
  via load_species_file, which also pins instrument='QE', NCE=30), then kept ONLY if the bare
  sequence convert_mod(SEQ)[0] is in the clean set train_sequences_clean_trainfold.txt (D8:
  filter DOWN only, no new seqs, same convert_mod convention). Random UNBIASED cap of
  --max_per_file per file (random.sample, seeded).
- Base = v1.4.1 + v3 installed model (ModelManager.load_installed_models). Same as baseline.
- FIXED normalization: target built by matching RAW intensities to the grid, then APD's own
  normalize_fragment_intensities (grid-max) -- element-wise identical to the verified run_all.py
  fix (verify_norm_fix.py V2 = 0.000). instrument=QE.
- val_fold (6 held-out species) used for epoch selection. PXD010613 is NOT touched here.
- Epoch selection: for each candidate in --epochs_candidates, fine-tune FRESH from pretrained
  for that many epochs (proper warmup+cosine schedule), eval val, pick best by val Spearman
  obs-top-N. ALL metrics printed per candidate (not just Spearman).

Output: --out_dir/epoch_<E>/ms2_finetuned.pth per candidate + a best pointer + val_selection.json.
Final PXD010613 eval is a SEPARATE step (eval_local_zeroshot_baseline.py --finetuned_model).
"""

import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import match_peaks_to_fragments, angular_similarity, pearson_corr
from eval_bacterial_hcd import load_species_file           # QE loader + filters
from eval_koina_bacterial import compute_fragment_mz, compute_gt_z1
from eval_local_zeroshot_baseline import spearman_obs_topN, cosine, summarize
from peptdeep.model.ms2 import normalize_fragment_intensities


def load_fold(list_file, clean_set, max_per_file, seed, limit_files=0):
    paths = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
    if limit_files:
        paths = paths[:limit_files]
    rng = random.Random(seed)
    rows = []
    for p in paths:
        r = load_species_file(Path(p), 10 ** 12)          # ALL spectra, QE, filtered
        if clean_set is not None:
            r = [x for x in r if x['sequence'] in clean_set]   # D8 leakage-clean, same convention
        if max_per_file and len(r) > max_per_file:
            r = rng.sample(r, max_per_file)               # uniform unbiased sample
        rows.extend(r)
    return rows


def build_fixed_target(model_mgr, rows):
    """Predict grid layout, match RAW intensities, then grid-max normalize (APD convention)."""
    pin = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in rows])
    pin['_orig_idx'] = range(len(rows))
    res = model_mgr.predict_all(pin, predict_items=['ms2'])
    pdf = res['precursor_df']
    if '_orig_idx' in pdf.columns:
        pdf = pdf.sort_values('_orig_idx').reset_index(drop=True)
    fmz = res['fragment_mz_df']
    fint = res['fragment_intensity_df']
    ion_cols = fmz.columns.tolist()
    target = np.zeros_like(fint.values, dtype=float)
    for idx, r in enumerate(rows):
        start = int(pdf.iloc[idx]['frag_start_idx'])
        stop = int(pdf.iloc[idx]['frag_stop_idx'])
        fmzs = fmz.iloc[start:stop].values
        target[start:stop, :] = match_peaks_to_fragments(fmzs, r['exp_mz'], r['exp_intensity'])
    target_df = pd.DataFrame(target, columns=ion_cols)
    normalize_fragment_intensities(pdf, target_df)        # grid-max == verified run_all fix
    return pdf, target_df


def eval_fold(model_mgr, rows, topn):
    pin = pd.DataFrame([{
        'sequence': r['sequence'], 'mods': r['mods'], 'mod_sites': r['mod_sites'],
        'charge': r['charge'], 'nce': r['nce'], 'instrument': r['instrument'],
    } for r in rows])
    pin['_orig_idx'] = range(len(rows))
    res = model_mgr.predict_all(pin, predict_items=['ms2'])
    pdf = res['precursor_df']
    if '_orig_idx' in pdf.columns:
        pdf = pdf.sort_values('_orig_idx').reset_index(drop=True)
    fdf = res['fragment_intensity_df']
    recs = []
    for idx, r in enumerate(rows):
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
        recs.append(('val', cosine(pred, gt), float(angular_similarity(pred, gt)),
                     float(pcc) if pcc == pcc else np.nan, spearman_obs_topN(gt, pred, topn)))
    return summarize(recs, topn)


def save_ms2(model_mgr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ms2 = model_mgr.ms2_model
    try:
        if hasattr(ms2, 'save'):
            ms2.save(str(path))
        else:
            import torch
            torch.save(ms2.model.state_dict(), str(path))
    except Exception:
        import torch
        torch.save(ms2.model.state_dict(), str(path))


def main():
    ap = argparse.ArgumentParser(description='T3 fine-tune with val_fold epoch selection.')
    ap.add_argument('--train_fold_list', required=True)
    ap.add_argument('--val_fold_list', required=True)
    ap.add_argument('--clean_seqs', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_per_file', type=int, default=2000)
    ap.add_argument('--val_max_per_file', type=int, default=1000)
    ap.add_argument('--epochs_candidates', default='3,6,10')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--topn', type=int, default=7)
    ap.add_argument('--limit_train_files', type=int, default=0, help='smoke: cap #train files')
    ap.add_argument('--verbose_each_epoch', action='store_true')
    args = ap.parse_args()

    from peptdeep.pretrained_models import ModelManager

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [int(x) for x in args.epochs_candidates.split(',') if x.strip()]

    clean_set = {s.strip() for s in Path(args.clean_seqs).read_text().splitlines() if s.strip()}
    print(f"clean_set: {len(clean_set)} sequences", flush=True)

    train_rows = load_fold(args.train_fold_list, clean_set, args.max_per_file, args.seed,
                           args.limit_train_files)
    val_rows = load_fold(args.val_fold_list, None, args.val_max_per_file, args.seed,
                         args.limit_train_files)
    print(f"train rows: {len(train_rows)}   val rows: {len(val_rows)}", flush=True)

    # Build the FIXED-normalization target ONCE (layout is model-independent).
    print("Building fixed-normalization (grid-max) training target...", flush=True)
    mm0 = ModelManager()
    mm0.load_installed_models()
    pdf, target_df = build_fixed_target(mm0, train_rows)
    gmax_ok = all(
        (target_df.iloc[int(pdf.iloc[i]['frag_start_idx']):int(pdf.iloc[i]['frag_stop_idx'])]
         .values.max() in (0.0,)) or
        abs(target_df.iloc[int(pdf.iloc[i]['frag_start_idx']):int(pdf.iloc[i]['frag_stop_idx'])]
            .values.max() - 1.0) < 1e-6
        for i in range(min(len(pdf), 50)))
    print(f"  target grid-max==1 spot-check (first 50): {'OK' if gmax_ok else 'FAIL'}", flush=True)

    results = {}
    for E in candidates:
        print(f"\n=== candidate epochs={E}  (lr={args.lr} batch={args.batch} warmup={args.warmup}) ===",
              flush=True)
        mm = ModelManager()
        mm.load_installed_models()                        # fresh pretrained each candidate
        mm.ms2_model.train(
            pdf, fragment_intensity_df=target_df,
            epoch=E, warmup_epoch=args.warmup, lr=args.lr, batch_size=args.batch,
            verbose=True, verbose_each_epoch=args.verbose_each_epoch,
        )
        save_ms2(mm, out_dir / f"epoch_{E}" / "ms2_finetuned.pth")
        m = eval_fold(mm, val_rows, args.topn)
        results[E] = m
        print(f"  VAL epochs={E}: Spearman={m['spearman_mean']:.4f}  SA={m['spectral_angle_mean']:.4f}  "
              f"cos={m['cosine_mean']:.4f}  PCC={m['pcc_mean']:.4f}  PCC90={m['pcc90']*100:.1f}%  "
              f"n={m['n']}", flush=True)

    # ── Selection (by val Spearman obs-topN) + all-metrics table ────────────────
    best_E = max(results, key=lambda e: results[e]['spearman_mean'])
    print("\n" + "=" * 78)
    print("T3 VAL-FOLD EPOCH SELECTION  (headline = Spearman obs-top%d, PROVISIONAL)" % args.topn)
    print(f"{'epochs':>7} {'Spearman':>9} {'SpecAngle':>10} {'cosine':>8} {'PCC':>7} {'PCC90':>7} {'n':>7}")
    for E in candidates:
        m = results[E]
        star = '  <== best' if E == best_E else ''
        print(f"{E:>7} {m['spearman_mean']:>9.4f} {m['spectral_angle_mean']:>10.4f} "
              f"{m['cosine_mean']:>8.4f} {m['pcc_mean']:>7.4f} {m['pcc90']*100:>6.1f}% {m['n']:>7}{star}")
    print("=" * 78)
    print(f"SELECTED epochs = {best_E}  ->  {out_dir / f'epoch_{best_E}' / 'ms2_finetuned.pth'}")
    with open(out_dir / 'val_selection.json', 'w') as f:
        json.dump({'candidates': candidates, 'selected_epochs': best_E,
                   'lr': args.lr, 'batch': args.batch, 'warmup': args.warmup,
                   'max_per_file': args.max_per_file, 'seed': args.seed,
                   'n_train': len(train_rows), 'n_val': len(val_rows),
                   'topn': args.topn, 'val_results': {str(k): v for k, v in results.items()}},
                  f, indent=2)
    print("NOTE: final PXD010613 eval is a SEPARATE step with the selected checkpoint.")


if __name__ == '__main__':
    main()
