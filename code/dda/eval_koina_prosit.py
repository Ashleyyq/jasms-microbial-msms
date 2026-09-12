"""
Evaluate Koina API model(s) on fixed NIST human test set. Uses NIST annotation-based GT.
Supports multiple models via --models. Auto-detects each model's required input format
(3-input Prosit, 4-input AlphaPeptDeep with instrument_types, or 2-input CID without CE).
Uses API annotation output for b/y z1 extraction when available; fallback to fixed offset.
Saves results/{model_name}_baseline.json per model.
"""
import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msp_utils import parse_msp_file, extract_peptide_info, angular_similarity


def pearson_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

KOINA_BASE = "https://koina.proteomicsdb.org/v2/models"
BATCH_SIZE = 50

# Probe constants (single peptide, unmodified)
_PROBE_SEQS = ["PEPTIDEK"]
_PROBE_CHARGES = [2]
_PROBE_NCES = [30.0]
_PROBE_INSTRUMENTS = ["QE"]

# ── Payload builders ──────────────────────────────────────────────────────────

def build_payload_prosit(seqs, charges, nces, instruments=None):
    """Standard 3-input Prosit: sequence, charge, collision_energy."""
    return {
        "id": "eval",
        "inputs": [
            {"name": "peptide_sequences", "shape": [len(seqs), 1], "datatype": "BYTES", "data": seqs},
            {"name": "precursor_charges", "shape": [len(charges), 1], "datatype": "INT32", "data": charges},
            {"name": "collision_energies", "shape": [len(nces), 1], "datatype": "FP32", "data": nces},
        ]
    }


def build_payload_alphapeptdeep(seqs, charges, nces, instruments=None):
    """4-input AlphaPeptDeep: adds instrument_types to standard Prosit inputs."""
    if instruments is None:
        instruments = ["QE"] * len(seqs)
    return {
        "id": "eval",
        "inputs": [
            {"name": "peptide_sequences", "shape": [len(seqs), 1], "datatype": "BYTES", "data": seqs},
            {"name": "precursor_charges", "shape": [len(charges), 1], "datatype": "INT32", "data": charges},
            {"name": "collision_energies", "shape": [len(nces), 1], "datatype": "FP32", "data": nces},
            {"name": "instrument_types", "shape": [len(instruments), 1], "datatype": "BYTES", "data": instruments},
        ]
    }


def build_payload_cid(seqs, charges, nces, instruments=None):
    """2-input CID: sequence and charge only (no collision energy)."""
    return {
        "id": "eval",
        "inputs": [
            {"name": "peptide_sequences", "shape": [len(seqs), 1], "datatype": "BYTES", "data": seqs},
            {"name": "precursor_charges", "shape": [len(charges), 1], "datatype": "INT32", "data": charges},
        ]
    }


def build_payload_prosit_fragtype(seqs, charges, nces, fragmentation_types=None):
    """4-input Prosit with fragmentation_types: sequence, charge, collision_energy, fragmentation_type.
    Used by models like Prosit_2025_intensity_40PTM that take 'fragmentation_types' instead of
    'instrument_types'. Accepted values: 'HCD', 'CID'."""
    if fragmentation_types is None:
        fragmentation_types = ["HCD"] * len(seqs)
    return {
        "id": "eval",
        "inputs": [
            {"name": "peptide_sequences",  "shape": [len(seqs), 1],              "datatype": "BYTES", "data": seqs},
            {"name": "precursor_charges",  "shape": [len(charges), 1],           "datatype": "INT32", "data": charges},
            {"name": "collision_energies", "shape": [len(nces), 1],              "datatype": "FP32",  "data": nces},
            {"name": "fragmentation_types","shape": [len(fragmentation_types), 1],"datatype": "BYTES", "data": fragmentation_types},
        ]
    }


# Ordered probe attempts (tried in sequence until one succeeds):
#   1. Standard Prosit 3-input
#   2. AlphaPeptDeep 4-input (instrument_types)
#   3. Prosit_2025-style 4-input (fragmentation_types)  ← new
#   4. CID 2-input (no collision energy)
_PROBE_FORMATS = [
    (build_payload_prosit,          3),
    (build_payload_alphapeptdeep,   4),
    (build_payload_prosit_fragtype, 4),
    (build_payload_cid,             2),
]

# ── Ion extraction helpers ────────────────────────────────────────────────────

# Fallback: Prosit 174 = 29 positions × 6 [y+, y++, y+++, b+, b++, b+++]
#   y+ at index k*6+0, b+ at index k*6+3  (k = 0..28)
def extract_z1_from_intensities_fallback(intensities_174, n_ions):
    """Use hardcoded offset when annotation is not available."""
    p = np.array(intensities_174, dtype=np.float32)
    b1 = np.array([p[k * 6 + 3] for k in range(n_ions)])
    y1 = np.array([p[k * 6 + 0] for k in range(n_ions)])
    return np.concatenate([b1, y1])


# Match "b3+1", "y5+1", "b2^1", "y1^1" etc. for z=1
RE_B_Z1 = re.compile(r'^b(\d+)[+^]1$', re.IGNORECASE)
RE_Y_Z1 = re.compile(r'^y(\d+)[+^]1$', re.IGNORECASE)


def extract_z1_from_annotation(intensities, annotations, n_ions):
    """
    Build pred_b_z1 and pred_y_z1 from Koina annotation labels.
    annotations: list of str, same length as intensities (typically 174).
    Fills only positions with a matching b/y z=1 annotation; rest remain 0.

    Koina returns "b3+1", "y5+1" (plus sign). NIST uses "b3^2" (caret).
    The regexes accept both [+^] for z=1.
    """
    pred_b = np.zeros(n_ions)
    pred_y = np.zeros(n_ions)
    n = min(len(intensities), len(annotations))
    for i in range(n):
        ann = str(annotations[i]).strip().lower()
        val = float(intensities[i])
        mb = RE_B_Z1.match(ann)
        my = RE_Y_Z1.match(ann)
        if mb:
            pos = int(mb.group(1)) - 1
            if 0 <= pos < n_ions:
                pred_b[pos] = max(pred_b[pos], val)
        elif my:
            pos = int(my.group(1)) - 1
            if 0 <= pos < n_ions:
                pred_y[pos] = max(pred_y[pos], val)
    return np.concatenate([pred_b, pred_y])


# ── Koina communication ───────────────────────────────────────────────────────

def probe_model(model_name):
    """
    Probe model with a single peptide to detect input format and output structure.
    Tries formats in order: 3-input Prosit → 4-input AlphaPeptDeep → 2-input CID.
    Returns (probe_response, use_annotation, payload_builder, error_str).
    payload_builder is None on failure.
    """
    url = f"{KOINA_BASE}/{model_name}/infer"
    last_err = "No probe attempted"

    for builder, n_inputs in _PROBE_FORMATS:
        payload = builder(_PROBE_SEQS, _PROBE_CHARGES, _PROBE_NCES, _PROBE_INSTRUMENTS)
        payload["id"] = "probe"
        try:
            r = requests.post(url, json=payload, timeout=30)
        except Exception as e:
            return None, False, None, str(e)

        if r.status_code == 200:
            data = r.json()
            out_names = [o.get("name", "") for o in data.get("outputs", [])]
            has_ann = any(n in out_names for n in ("annotation", "annotations", "fragment_annotation"))
            print(f"  Input format: {builder.__name__} ({n_inputs} inputs), annotation: {has_ann}")
            return data, has_ann, builder, None

        last_err = f"({n_inputs}-input) status {r.status_code}: {r.text[:300]}"
        # Continue trying next format for any 400 error (bad request / wrong input format).
        # Non-400 errors (network, auth, server-side) are not retried.
        if r.status_code != 400:
            break

    return None, False, None, last_err


def call_koina(model_name, payload):
    url = f"{KOINA_BASE}/{model_name}/infer"
    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            return None, r.text
        return r.json(), None
    except Exception as e:
        return None, str(e)


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_model(model_name, df, gt_z1_list, use_annotation, payload_builder,
              fragmentation_type="HCD"):
    """Run model over all batches and return (as_list, pcc_list)."""
    n = len(df)
    seqs        = df['sequence'].astype(str).tolist()
    charges     = df['charge'].astype(int).tolist()
    nces        = df['nce'].astype(float).tolist()
    instruments = df['instrument'].astype(str).tolist() if 'instrument' in df.columns else ["QE"] * n

    as_list  = []
    pcc_list = []
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        # Route to the correct builder signature.
        # build_payload_prosit_fragtype expects fragmentation_types, not instruments.
        builder_name = payload_builder.__name__
        if builder_name == 'build_payload_prosit_fragtype':
            frag_types = [fragmentation_type] * (end - i)
            payload = payload_builder(seqs[i:end], charges[i:end],
                                      nces[i:end], frag_types)
        else:
            payload = payload_builder(seqs[i:end], charges[i:end], nces[i:end], instruments[i:end])
        res, err = call_koina(model_name, payload)
        if err:
            print(f"    Batch {i}-{end} error: {err[:200]}")
            continue

        outputs = {o["name"]: o for o in res.get("outputs", [])}

        # Locate intensity array
        int_data = None
        for name in ("intensities", "intensity"):
            if name in outputs:
                int_data = np.array(outputs[name]["data"], dtype=np.float32)
                break
        if int_data is None:
            for o in res.get("outputs", []):
                d = o.get("data")
                if d and len(d) == (end - i) * 174:
                    int_data = np.array(d, dtype=np.float32)
                    break
        if int_data is None:
            print(f"    Batch {i}-{end}: could not find intensity output, skipping.")
            continue

        # Locate annotation array (optional)
        ann_data = None
        if use_annotation:
            for name in ("annotation", "annotations", "fragment_annotation"):
                if name in outputs:
                    ann_data = outputs[name].get("data")
                    if ann_data:
                        break

        batch_n = end - i
        if int_data.size % batch_n != 0:
            print(f"    Batch {i}-{end}: output size {int_data.size} not divisible "
                  f"by {batch_n} peptides; skipping batch.")
            continue
        ions_per_pred = int_data.size // batch_n      # 174 for Prosit, may differ for other models
        preds = int_data.reshape(batch_n, ions_per_pred)

        for j in range(batch_n):
            n_ions = len(seqs[i + j]) - 1
            if use_annotation and ann_data and len(ann_data) >= (j + 1) * ions_per_pred:
                ann_slice = ann_data[j * ions_per_pred:(j + 1) * ions_per_pred]
                pred_z1 = extract_z1_from_annotation(preds[j], ann_slice, n_ions)
            else:
                pred_z1 = extract_z1_from_intensities_fallback(preds[j], n_ions)

            if np.max(pred_z1) > 0:
                pred_z1 = pred_z1 / np.max(pred_z1)
            gt_z1 = gt_z1_list[i + j].copy()
            if np.max(gt_z1) > 0:
                gt_z1 = gt_z1 / np.max(gt_z1)
            as_list.append(angular_similarity(pred_z1, gt_z1))
            pcc_list.append(pearson_corr(pred_z1, gt_z1))

    return as_list, pcc_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Koina models on NIST human test set (annotation-based GT).')
    parser.add_argument('--test_set_dir', default='results',
                        help='Directory containing test_set.csv and test_set.msp')
    parser.add_argument('--models', nargs='+',
                        default=[
                            'Prosit_2019_intensity',
                            'Prosit_2020_intensity_HCD',
                            'Prosit_2025_intensity_40PTM',
                            'AlphaPeptDeep_ms2_generic',
                            'Prosit_2020_intensity_CID',
                        ],
                        help='Koina model names to evaluate')
    parser.add_argument('--fragmentation_type', default='HCD',
                        choices=['HCD', 'CID'],
                        help='Fragmentation type for models that require it '
                             '(e.g. Prosit_2025_intensity_40PTM). '
                             'Default: HCD (human NIST data is HCD-acquired)')
    parser.add_argument('--out_dir', default='results',
                        help='Directory for output JSON files (default: results)')
    args = parser.parse_args()

    base = Path(args.test_set_dir)
    csv_path = base / 'test_set.csv'
    msp_path = base / 'test_set.msp'
    if not csv_path.exists() or not msp_path.exists():
        raise FileNotFoundError(
            f"Need {csv_path} and {msp_path}. Run prepare_test_set.py first.")

    df = pd.read_csv(csv_path)
    spectra = parse_msp_file(str(msp_path), max_spectra=None)
    if len(spectra) != len(df):
        raise ValueError(f"MSP has {len(spectra)} spectra but CSV has {len(df)} rows.")

    gt_z1_list = []
    for spec in spectra:
        info = extract_peptide_info(spec)
        if info is None:
            raise ValueError("Spectrum in test_set.msp failed extract_peptide_info.")
        gt_z1_list.append(np.concatenate([info['b_z1_ground_truth'], info['y_z1_ground_truth']]))

    results_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n--- Model: {model_name} ---")
        probe_data, use_annotation, payload_builder, err = probe_model(model_name)
        if probe_data is None:
            print(f"  Probe failed: {err}")
            print(f"  Skipping {model_name}.")
            continue

        as_list, pcc_list = run_model(model_name, df, gt_z1_list, use_annotation, payload_builder,
                                      fragmentation_type=args.fragmentation_type)
        if not as_list:
            print(f"  No results for {model_name}.")
            continue

        as_arr  = np.array(as_list)
        pcc_arr = np.array(pcc_list)
        pcc90   = float(np.mean(pcc_arr >= 0.9))
        out = {
            'model': model_name,
            'n_spectra': len(as_list),
            'metric': 'angular_similarity_z1_only',
            'mean_as': float(np.mean(as_arr)),
            'median_as': float(np.median(as_arr)),
            'std_as': float(np.std(as_arr)),
            'mean_pcc': float(np.mean(pcc_arr)),
            'median_pcc': float(np.median(pcc_arr)),
            'pcc90': pcc90,
        }
        out_path = results_dir / f"{model_name}_baseline.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"  AS:  mean={out['mean_as']:.4f} median={out['median_as']:.4f} std={out['std_as']:.4f}")
        print(f"  PCC: mean={out['mean_pcc']:.4f} PCC90={pcc90*100:.1f}% -> {out_path}")


if __name__ == '__main__':
    main()
