"""
Shared MSP parsing and ground-truth extraction for NIST human spectral libraries.
Ground truth: NIST annotation-based (e.g. b2/-2.1ppm, y5^2). No m/z matching.
Used by prepare_test_set, eval_koina_prosit, eval_alphapeptdeep_baseline.
"""
import re
import numpy as np


def parse_msp_file(msp_path, max_spectra=None):
    """Parse NIST MSP spectral library, including peak annotations."""
    spectra = []
    peak_re = re.compile(r'^\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s*"?([^"]*)"?')

    with open(msp_path, 'r', encoding='utf-8') as f:
        cur = {}
        mzs, ints, annots = [], [], []
        in_peaks = False

        for line in f:
            line = line.rstrip('\n')
            if line.startswith('Name:'):
                if cur and mzs:
                    cur['mz'] = np.array(mzs, dtype='float64')
                    cur['intensity'] = np.array(ints, dtype='float64')
                    cur['annotation'] = np.array(annots, dtype='object')
                    spectra.append(cur)
                    if max_spectra and len(spectra) >= max_spectra:
                        break
                cur = {'Name': line.split(':', 1)[1].strip()}
                mzs, ints, annots = [], [], []
                in_peaks = False
            elif line.startswith('MW:'):
                cur['MW'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Comment:'):
                comment = line.split(':', 1)[1].strip()
                cur['Comment'] = comment
                for field in comment.split():
                    if '=' in field:
                        k, v = field.split('=', 1)
                        cur[k] = v
            elif line.startswith('Num peaks:'):
                in_peaks = True
            elif in_peaks and line.strip():
                m = peak_re.match(line)
                if m:
                    mzs.append(float(m.group(1)))
                    ints.append(float(m.group(2)))
                    annots.append(m.group(3).strip())
            elif not line.strip():
                in_peaks = False

        if cur and mzs:
            cur['mz'] = np.array(mzs, dtype='float64')
            cur['intensity'] = np.array(ints, dtype='float64')
            cur['annotation'] = np.array(annots, dtype='object')
            spectra.append(cur)

    return spectra


def extract_peptide_info(spectrum):
    """Extract sequence, charge, mods, and annotated b/y ground truth (z1 and z2). Returns None if invalid."""
    name = spectrum.get('Name', '')
    match = re.match(r'^([A-Z]+)/(\d+)', name)
    if not match:
        return None

    raw_seq = match.group(1)
    charge = int(match.group(2))
    if charge < 1 or charge > 6:
        return None
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(a in valid_aa for a in raw_seq):
        return None
    if len(raw_seq) < 7 or len(raw_seq) > 30:
        return None

    mods = []
    sites = []
    mods_str = spectrum.get('Mods', '0')
    if mods_str and mods_str != '0':
        mod_pattern = re.compile(r'\(([^)]+)\)')
        for m_str in mod_pattern.findall(mods_str):
            parts = m_str.split(',')
            if len(parts) >= 3:
                try:
                    pos = int(parts[0])
                    mod_name = parts[2]
                    site_idx = str(pos + 1)
                    if 'Oxidation' in mod_name or 'oxidation' in mod_name:
                        mods.append('Oxidation@M')
                        sites.append(site_idx)
                    elif 'Carbamidomethyl' in mod_name or 'CAM' in mod_name:
                        mods.append('Carbamidomethyl@C')
                        sites.append(site_idx)
                    elif 'Acetyl' in mod_name or 'acetyl' in mod_name:
                        if pos == 0:
                            mods.append('Acetyl@Protein_N-term')
                            sites.append('0')
                        else:
                            return None
                    else:
                        return None
                except ValueError:
                    return None

    nce_val = 30.0
    nce_str = spectrum.get('NCE', spectrum.get('HCD', spectrum.get('CE', '')))
    if nce_str:
        try:
            nce_val = float(re.sub(r'[^0-9.]', '', str(nce_str)))
        except ValueError:
            pass

    n_ions = len(raw_seq) - 1
    b_z1_gt = np.zeros(n_ions)
    b_z2_gt = np.zeros(n_ions)
    y_z1_gt = np.zeros(n_ions)
    y_z2_gt = np.zeros(n_ions)
    annotations = spectrum.get('annotation', np.array([]))
    intensities = spectrum.get('intensity', np.array([]))
    if len(intensities) > 0:
        norm_int = intensities / np.max(intensities)
    else:
        norm_int = intensities

    # NIST annotation: b2/-2.1ppm, y5^2, etc. Ion type, position (1-based), optional charge.
    ion_re = re.compile(r'^(b|y)(\d+)(?:\^(\d+))?')
    for annot, inten in zip(annotations, norm_int):
        for part in str(annot).split(','):
            clean_part = part.split('/')[0].strip()
            m = ion_re.match(clean_part)
            if m:
                ion_type = m.group(1)
                pos = int(m.group(2)) - 1
                ion_charge = int(m.group(3)) if m.group(3) else 1
                if 0 <= pos < n_ions:
                    if ion_type == 'b':
                        if ion_charge == 1:
                            b_z1_gt[pos] = max(b_z1_gt[pos], inten)
                        elif ion_charge == 2:
                            b_z2_gt[pos] = max(b_z2_gt[pos], inten)
                    elif ion_type == 'y':
                        if ion_charge == 1:
                            y_z1_gt[pos] = max(y_z1_gt[pos], inten)
                        elif ion_charge == 2:
                            y_z2_gt[pos] = max(y_z2_gt[pos], inten)

    return {
        'sequence': raw_seq,
        'mods': ';'.join(mods),
        'mod_sites': ';'.join(sites),
        'charge': charge,
        'nce': nce_val,
        'instrument': None,
        'b_z1_ground_truth': b_z1_gt,
        'b_z2_ground_truth': b_z2_gt,
        'y_z1_ground_truth': y_z1_gt,
        'y_z2_ground_truth': y_z2_gt,
    }


def angular_similarity(pred, exp):
    """AS = 1 - (2/pi)*arccos(cosine_sim). Zero vector vs zero = 0."""
    mask = (pred > 1e-6) | (exp > 1e-6)
    if mask.sum() < 2:
        return 0.0
    p, e = pred[mask], exp[mask]
    n1, n2 = np.linalg.norm(p), np.linalg.norm(e)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = np.clip(np.dot(p, e) / (n1 * n2), -1, 1)
    return float(1 - 2 * np.arccos(cos) / np.pi)


def count_b_y_annotations(info):
    """Number of positions with at least one b or y (z1 or z2) annotation."""
    b = np.maximum(info['b_z1_ground_truth'], info['b_z2_ground_truth'])
    y = np.maximum(info['y_z1_ground_truth'], info['y_z2_ground_truth'])
    return int(np.sum(b > 1e-6) + np.sum(y > 1e-6))
