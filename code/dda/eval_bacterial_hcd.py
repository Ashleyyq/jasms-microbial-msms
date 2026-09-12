"""
eval_bacterial_hcd.py
=====================
HCD bacterial evaluation (PXD010613, 4 species) using Koina public API.
Mirrors eval_bacterial_112sp.py with these differences only:
  - DEFAULT_MODELS: HCD + APD only (no CID — these are HCD-acquired spectra)
  - instrument='QE' (Q Exactive, matching PXD010613 acquisition)
  - Default data_dir: ~/peptdeep_project/data/bacteria_hcd
  - Default out_dir:  ~/peptdeep_project/results/eval_hcd
  - Output filenames: *_hcd_result.json, summary_hcd.json

Purpose: close the "symmetry matrix" — CID models dominate on CID bacterial data
(Prosit_2020_CID AS=0.7445, PCC90=52.4% on 112sp). Now evaluate HCD models on
HCD-acquired bacterial data to show the reverse.

Dataset: PXD010613 (Kaiko et al.) — 4 organisms, Q Exactive / HCD / NCE=30.
Species: Enterococcus faecalis, Akkermansia muciniphila,
         Halanaerobium congolense, Caulobacter crescentus.

Usage (login node, inside screen to survive disconnection):
  screen -S eval_hcd
  conda activate peptdeep
  cd ~/peptdeep_project
  python scripts/eval_bacterial_hcd.py

Resume after interruption (same command — auto-skips completed species):
  python scripts/eval_bacterial_hcd.py

Fallback if login node OOMs:
  srun --partition=short --mem=16G python scripts/eval_bacterial_hcd.py
"""

import argparse
import gc
import json
import sys
import time
import numpy as np
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Metrics from run_all.py (no data loading here — we load per-file below)
from run_all import parse_mgf_file, convert_mod, angular_similarity, pearson_corr

# GT construction helpers from eval_koina_bacterial.py
from eval_koina_bacterial import (
    compute_fragment_mz,
    compute_gt_z1,
    TOLERANCE,
)

# Koina helpers from eval_koina_prosit.py
from eval_koina_prosit import (
    KOINA_BASE,
    BATCH_SIZE,
    extract_z1_from_annotation,
    probe_model,
)

# ── Constants ──────────────────────────────────────────────────────────────────

INTER_BATCH_SLEEP = 0.5   # seconds between successful batches
MAX_RETRIES       = 5     # max retries on 429 / 5xx
RETRY_BASE_SLEEP  = 1.0   # seconds × 2^attempt

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_MODELS = [
    'Prosit_2020_intensity_HCD',
    'AlphaPeptDeep_ms2_generic',
]


# ── Per-file data loading (mirrors load_data() filtering in run_all.py) ────────

def load_species_file(mgf_path: Path, max_spectra: int) -> list:
    """
    Load and filter one MGF file.  Mirrors the filtering logic in run_all.load_data()
    exactly: valid AA, length 7–30, no Unknown mods, charge 1–6.
    Returns list of row dicts (same schema as load_data rows).
    Peak arrays are numpy float64 — caller must del the list to free memory.
    """
    spectra = parse_mgf_file(str(mgf_path), max_spectra)
    rows = []
    for s in spectra:
        seq, mods, sites = convert_mod(s.get('SEQ', ''))
        if not seq or not all(a in VALID_AA for a in seq):
            continue
        if len(seq) < 7 or len(seq) > 30:
            continue
        if 'Unknown' in mods:
            continue
        raw_ch = s.get('CHARGE', '2+').replace('+', '').replace('-', '')
        try:
            ch = int(raw_ch)
        except ValueError:
            continue
        if ch < 1 or ch > 6:
            continue
        rows.append({
            'sequence':     seq,
            'mods':         mods,
            'mod_sites':    sites,
            'charge':       ch,
            'nce':          30.0,
            'instrument':   'QE',
            'exp_mz':       s['mz'],
            'exp_intensity': s['intensity'],
            'source':       s['_file'],
        })
    return rows


# ── Retry-aware Koina call ─────────────────────────────────────────────────────

def call_koina_with_retry(model_name: str, payload: dict):
    """
    POST to Koina with exponential backoff on 429 and 5xx.
    Returns (response_json, error_str).  error_str is None on success.
    400 is NOT retried (bad request = wrong payload format).
    """
    url = f"{KOINA_BASE}/{model_name}/infer"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=60)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_SLEEP * (2 ** attempt)
                print(f"      [retry {attempt+1}/{MAX_RETRIES}] network error: {e}; "
                      f"sleep {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            return None, str(e)

        if r.status_code == 200:
            return r.json(), None

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_SLEEP * (2 ** attempt)
                print(f"      [retry {attempt+1}/{MAX_RETRIES}] HTTP {r.status_code}; "
                      f"sleep {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue

        return None, f"HTTP {r.status_code}: {r.text[:300]}"

    return None, f"All {MAX_RETRIES} retries exhausted"


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path) -> dict:
    """Load {species: result_dict}.  Returns {} if file does not exist."""
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {len(data)} species already done ({ckpt_path.name})")
        return data
    return {}


def save_checkpoint(ckpt_path: Path, checkpoint: dict):
    """Atomically write checkpoint via .tmp rename."""
    tmp = ckpt_path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    tmp.rename(ckpt_path)


# ── Per-species evaluation ─────────────────────────────────────────────────────

def eval_species(model_name: str, test_rows: list,
                 use_annotation: bool, payload_builder) -> dict:
    """
    Evaluate one Koina model on one species' test rows.
    Returns {mean_as, median_as, std_as, mean_pcc, pcc90, pcc75, n, n_fail}.
    pcc90 = fraction of spectra with PCC >= 0.9 (spectrum-level, not species-level).
    """
    n           = len(test_rows)
    seqs        = [r['sequence']   for r in test_rows]
    charges     = [r['charge']     for r in test_rows]
    nces        = [r['nce']        for r in test_rows]
    instruments = [r['instrument'] for r in test_rows]

    # Pre-compute GT for all test rows of this species (cheap)
    gt_list = []
    for r in test_rows:
        b_mz, y_mz = compute_fragment_mz(r['sequence'], r['mods'], r['mod_sites'])
        gt_list.append(compute_gt_z1(r, b_mz, y_mz))

    as_list  = []
    pcc_list = []
    n_fail   = 0

    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)

        builder_name = payload_builder.__name__
        if builder_name == 'build_payload_prosit_fragtype':
            frag_types = ['CID'] * (end - i)
            payload = payload_builder(seqs[i:end], charges[i:end], nces[i:end], frag_types)
        elif builder_name == 'build_payload_cid':
            payload = payload_builder(seqs[i:end], charges[i:end], nces[i:end], None)
        else:
            payload = payload_builder(seqs[i:end], charges[i:end], nces[i:end],
                                      instruments[i:end])

        res, err = call_koina_with_retry(model_name, payload)
        if err:
            print(f"      batch {i}–{end} FAILED: {err[:150]}", flush=True)
            for j in range(end - i):
                n_ions  = len(seqs[i + j]) - 1
                pred_z1 = np.zeros(n_ions * 2)
                gt = gt_list[i + j]
                k  = min(len(pred_z1), len(gt))
                as_list.append(angular_similarity(pred_z1[:k], gt[:k]))
                pcc_list.append(pearson_corr(pred_z1[:k], gt[:k]))
            n_fail += (end - i)
            time.sleep(INTER_BATCH_SLEEP)
            continue

        outputs  = {o['name']: o for o in res.get('outputs', [])}
        int_data = None
        for name in ('intensities', 'intensity'):
            if name in outputs:
                int_data = np.array(outputs[name]['data'], dtype=np.float32)
                break
        if int_data is None:
            print(f"      batch {i}–{end}: no intensity field, skipping.", flush=True)
            time.sleep(INTER_BATCH_SLEEP)
            continue

        # Annotation required — no hardcoded fallback
        ann_data = None
        for name in ('annotation', 'annotations', 'fragment_annotation'):
            if name in outputs:
                ann_data = outputs[name].get('data')
                if ann_data:
                    break
        if ann_data is None:
            print(f"      batch {i}–{end}: WARNING — annotation absent; "
                  f"scoring batch as zero", flush=True)

        batch_n = end - i
        if int_data.size % batch_n != 0:
            print(f"      batch {i}–{end}: size {int_data.size} not divisible by "
                  f"{batch_n}, skipping.", flush=True)
            time.sleep(INTER_BATCH_SLEEP)
            continue

        ions_per_pred = int_data.size // batch_n
        preds         = int_data.reshape(batch_n, ions_per_pred)

        for j in range(batch_n):
            n_ions = len(seqs[i + j]) - 1
            if ann_data and len(ann_data) >= (j + 1) * ions_per_pred:
                ann_slice = ann_data[j * ions_per_pred:(j + 1) * ions_per_pred]
                pred_z1   = extract_z1_from_annotation(preds[j], ann_slice, n_ions)
            else:
                # annotation absent — score as zero, never use hardcoded offset
                pred_z1 = np.zeros(n_ions * 2)

            if np.max(pred_z1) > 0:
                pred_z1 = pred_z1 / np.max(pred_z1)

            gt = gt_list[i + j]
            k  = min(len(pred_z1), len(gt))
            as_list.append(angular_similarity(pred_z1[:k], gt[:k]))
            pcc_list.append(pearson_corr(pred_z1[:k], gt[:k]))

        time.sleep(INTER_BATCH_SLEEP)

    as_arr  = np.array(as_list)
    pcc_arr = np.array(pcc_list)
    return {
        'mean_as':   float(np.mean(as_arr)),
        'median_as': float(np.median(as_arr)),
        'std_as':    float(np.std(as_arr)),
        'mean_pcc':  float(np.mean(pcc_arr)),
        'pcc90':     float(np.mean(pcc_arr >= 0.9)),
        'pcc75':     float(np.mean(pcc_arr >= 0.75)),
        'n':         int(len(as_list)),
        'n_fail':    int(n_fail),
    }


# ── Aggregate checkpoint → summary dict ───────────────────────────────────────

def aggregate(checkpoint: dict, species_list: list) -> dict:
    """Weighted aggregation of per-species metrics (weights = n spectra)."""
    sp_ns    = np.array([checkpoint[sp]['n']        for sp in species_list], dtype=float)
    sp_as    = np.array([checkpoint[sp]['mean_as']  for sp in species_list])
    sp_pcc   = np.array([checkpoint[sp]['mean_pcc'] for sp in species_list])
    sp_pcc90 = np.array([checkpoint[sp]['pcc90']    for sp in species_list])
    sp_pcc75 = np.array([checkpoint[sp]['pcc75']    for sp in species_list])
    w        = sp_ns / sp_ns.sum()
    return {
        'n_species':      len(species_list),
        'n_test_spectra': int(sp_ns.sum()),
        'n_failed':       int(sum(checkpoint[sp]['n_fail'] for sp in species_list)),
        'mean_as':        float(np.dot(w, sp_as)),
        'equal_mean_as':  float(np.mean(sp_as)),
        'mean_pcc':       float(np.dot(w, sp_pcc)),
        'pcc90':          float(np.dot(w, sp_pcc90)),
        'pcc75':          float(np.dot(w, sp_pcc75)),
        'per_species':    {sp: checkpoint[sp] for sp in species_list},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='HCD bacterial evaluation — PXD010613, 4 species (Koina API, low-memory, with checkpointing).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--data_dir',    default='~/peptdeep_project/data/bacteria_hcd')
    ap.add_argument('--max_spectra', type=int, default=5000,
                    help='Max spectra per MGF file (default: 5000)')
    ap.add_argument('--test_ratio',  type=float, default=0.2)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--out_dir',     default='~/peptdeep_project/results/eval_hcd')
    ap.add_argument('--models',      nargs='*', default=DEFAULT_MODELS)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir  = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover MGF files ────────────────────────────────────────────────────
    mgf_files = sorted(data_dir.glob('*.mgf'))
    if not mgf_files:
        sys.exit(f"ERROR: no .mgf files found in {data_dir}")
    species_list = [f.stem for f in mgf_files]
    n_sp = len(species_list)
    print(f"Found {n_sp} MGF files in {data_dir}")
    print(f"Settings: max_spectra={args.max_spectra}  test_ratio={args.test_ratio}  seed={args.seed}")
    print(f"Memory: loading one file at a time (~{args.max_spectra} rows peak)")

    # ── Probe models once at startup ──────────────────────────────────────────
    print("\nProbing Koina models ...")
    model_configs = {}
    for model_name in args.models:
        probe_data, use_ann, builder, err = probe_model(model_name)
        if probe_data is None:
            print(f"  [{model_name}] PROBE FAILED: {err}  — skipping")
        else:
            model_configs[model_name] = (use_ann, builder)
            print(f"  [{model_name}]  annotation={use_ann}  builder={builder.__name__}")
    if not model_configs:
        sys.exit("ERROR: all model probes failed. Check network / Koina URL.")

    # ── Load checkpoints ──────────────────────────────────────────────────────
    print()
    checkpoints = {}
    for model_name in model_configs:
        ckpt_path = out_dir / f'checkpoint_{model_name}.json'
        checkpoints[model_name] = load_checkpoint(ckpt_path)

    # ── Species-first main loop (one file in memory at a time) ────────────────
    total_species_done = {m: len(checkpoints[m]) for m in model_configs}
    print(f"\nStarting evaluation (species-first loop):")
    for m, n_done in total_species_done.items():
        print(f"  {m}: {n_done}/{n_sp} already done")

    for file_idx, (mgf_path, species) in enumerate(zip(mgf_files, species_list)):

        # Check if all models already have this species
        models_needed = [m for m in model_configs if species not in checkpoints[m]]
        if not models_needed:
            continue

        # Load this species file (only when needed)
        print(f"\n[{file_idx+1:>3d}/{n_sp}] {species}", flush=True)
        rows = load_species_file(mgf_path, args.max_spectra)
        if not rows:
            print(f"  WARNING: 0 valid rows, skipping.", flush=True)
            continue

        # Per-species deterministic split (seed derived from position for stability)
        np.random.seed(args.seed + file_idx)
        perm   = np.random.permutation(len(rows))
        n_test = max(1, int(len(rows) * args.test_ratio))
        test_rows = [rows[i] for i in perm[:n_test]]
        print(f"  rows={len(rows)}  test={len(test_rows)}", flush=True)

        # Evaluate each model that still needs this species
        for model_name in models_needed:
            use_ann, builder = model_configs[model_name]
            print(f"  [{model_name}]", end='  ', flush=True)
            t0     = time.time()
            result = eval_species(model_name, test_rows, use_ann, builder)
            elapsed = time.time() - t0

            checkpoints[model_name][species] = result
            ckpt_path = out_dir / f'checkpoint_{model_name}.json'
            save_checkpoint(ckpt_path, checkpoints[model_name])

            n_done = len(checkpoints[model_name])
            print(f"AS={result['mean_as']:.4f}  PCC={result['mean_pcc']:.4f}  "
                  f"PCC90={result['pcc90']*100:.1f}%  "
                  f"n={result['n']}  fail={result['n_fail']}  "
                  f"{elapsed:.1f}s  [{n_done}/{n_sp}]",
                  flush=True)

        # Free memory — critical for login node stability
        del rows, test_rows
        gc.collect()

    # ── Aggregate and write final outputs ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}")

    all_model_results = {}

    for model_name in model_configs:
        checkpoint = checkpoints[model_name]
        completed  = [sp for sp in species_list if sp in checkpoint]
        missing    = [sp for sp in species_list if sp not in checkpoint]
        if missing:
            print(f"  [{model_name}] WARNING: {len(missing)} species missing from checkpoint — "
                  f"run again to complete: {missing[:5]}")
        if not completed:
            continue

        agg = aggregate(checkpoint, completed)
        agg['model']       = model_name
        agg['gt_method']   = f'theoretical_mz_{TOLERANCE}Da_tolerance_z1_only'
        agg['max_spectra'] = args.max_spectra
        agg['test_ratio']  = args.test_ratio
        agg['seed']        = args.seed
        agg['n_missing']   = len(missing)
        all_model_results[model_name] = agg

        # Per-model JSON
        result_path = out_dir / f'{model_name}_hcd_result.json'
        with open(result_path, 'w') as f:
            json.dump(agg, f, indent=2)

        # Per-species CSV (sorted by mean_as descending)
        csv_path = out_dir / f'{model_name}_per_species.csv'
        with open(csv_path, 'w') as f:
            f.write('species,mean_as,median_as,std_as,mean_pcc,pcc90,pcc75,n,n_fail\n')
            for sp in sorted(completed, key=lambda s: checkpoint[s]['mean_as'], reverse=True):
                r = checkpoint[sp]
                f.write(f"{sp},{r['mean_as']:.4f},{r['median_as']:.4f},"
                        f"{r['std_as']:.4f},{r['mean_pcc']:.4f},"
                        f"{r['pcc90']:.4f},{r['pcc75']:.4f},"
                        f"{r['n']},{r['n_fail']}\n")

        print(f"\n  [{model_name}]  n_species={len(completed)}")
        print(f"    Mean AS  (weighted) = {agg['mean_as']:.4f}")
        print(f"    Mean PCC (weighted) = {agg['mean_pcc']:.4f}")
        print(f"    PCC90%   (weighted) = {agg['pcc90']*100:.1f}%")
        print(f"    N spectra           = {agg['n_test_spectra']}  (fail={agg['n_failed']})")
        print(f"    -> {result_path}")
        print(f"    -> {csv_path}")

    # ── Cross-model summary ───────────────────────────────────────────────────
    if not all_model_results:
        print("\nNo models produced results.")
        return

    summary_path = out_dir / 'summary_hcd.json'
    with open(summary_path, 'w') as f:
        json.dump(all_model_results, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY — HCD bacterial evaluation (PXD010613, 4 species)")
    print(f"  Unified GT: theoretical m/z ± {TOLERANCE} Da, z=1 b/y ions only")
    print(f"  Metrics weighted by n spectra per species")
    print(f"{'='*80}")
    hdr = (f"{'Model':<45s}  {'MeanAS':>7s}  {'MeanPCC':>8s}  "
           f"{'PCC90%':>7s}  {'PCC75%':>7s}  {'N':>7s}")
    print(hdr)
    print('-' * len(hdr))
    for name, r in sorted(all_model_results.items(),
                           key=lambda x: x[1]['mean_as'], reverse=True):
        print(f"{name:<45s}  {r['mean_as']:>7.4f}  {r['mean_pcc']:>8.4f}  "
              f"{r['pcc90']*100:>6.1f}%  {r['pcc75']*100:>6.1f}%  "
              f"{r['n_test_spectra']:>7d}")
    print(f"\nFull results -> {summary_path}")
    print("Per-species  -> {model}_per_species.csv")


if __name__ == '__main__':
    main()
