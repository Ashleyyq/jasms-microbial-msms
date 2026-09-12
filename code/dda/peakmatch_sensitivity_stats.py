#!/usr/bin/env python3
"""Pure numerical helpers for the JASMS peak-matching sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pick_peak(
    target_mz: float,
    exp_mz: np.ndarray,
    exp_intensity: np.ndarray,
    tolerance_da: float,
) -> tuple[float, float, int]:
    """Return nearest intensity, highest intensity, and candidate count."""
    distances = np.abs(exp_mz - target_mz)
    candidates = np.flatnonzero(distances <= tolerance_da)
    if not len(candidates):
        return 0.0, 0.0, 0

    nearest_index = candidates[np.argmin(distances[candidates])]
    candidate_intensity = exp_intensity[candidates]
    highest = np.max(candidate_intensity)
    # Resolve equal-intensity maxima by distance so the result is deterministic.
    highest_candidates = candidates[candidate_intensity == highest]
    highest_index = highest_candidates[np.argmin(distances[highest_candidates])]
    return (
        float(exp_intensity[nearest_index]),
        float(exp_intensity[highest_index]),
        int(len(candidates)),
    )


def match_ground_truth_pair(
    exp_mz,
    exp_intensity,
    b_z1_mz,
    y_z1_mz,
    tolerance_da: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build nearest-peak and highest-intensity b/y z=1 ground truths.

    Both returned vectors use ``[b1..b(n-1), y1..y(n-1)]`` ordering and are
    normalized to the experimental base peak before matching.
    """
    mz = np.asarray(exp_mz, dtype=np.float64)
    intensity = np.asarray(exp_intensity, dtype=np.float64)
    theoretical = np.concatenate(
        [np.asarray(b_z1_mz, dtype=np.float64), np.asarray(y_z1_mz, dtype=np.float64)]
    )
    nearest = np.zeros(len(theoretical), dtype=np.float64)
    highest = np.zeros(len(theoretical), dtype=np.float64)
    candidate_counts = np.zeros(len(theoretical), dtype=np.int32)

    if not len(intensity) or not np.isfinite(intensity).any():
        return nearest, highest, {
            "n_targets": int(len(theoretical)),
            "n_targets_with_match": 0,
            "n_multihit_targets": 0,
            "n_nearest_highest_different": 0,
            "max_abs_difference": 0.0,
        }
    maximum = float(np.nanmax(intensity))
    if maximum <= 0:
        return nearest, highest, {
            "n_targets": int(len(theoretical)),
            "n_targets_with_match": 0,
            "n_multihit_targets": 0,
            "n_nearest_highest_different": 0,
            "max_abs_difference": 0.0,
        }
    normalized = intensity / maximum

    for index, target in enumerate(theoretical):
        n_value, h_value, count = _pick_peak(
            float(target), mz, normalized, tolerance_da
        )
        nearest[index] = n_value
        highest[index] = h_value
        candidate_counts[index] = count

    difference = np.abs(nearest - highest)
    return nearest, highest, {
        "n_targets": int(len(theoretical)),
        "n_targets_with_match": int(np.sum(candidate_counts >= 1)),
        "n_multihit_targets": int(np.sum(candidate_counts >= 2)),
        "n_nearest_highest_different": int(np.sum(difference > 1e-12)),
        "max_abs_difference": float(np.max(difference)) if len(difference) else 0.0,
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    delta_column: str,
    cluster_column: str,
    reps: int,
    seed: int,
    block_size: int = 32,
    scale: float = 1.0,
) -> dict:
    """Bootstrap modified-precursor clusters for two explicit estimands.

    The ratio estimator retains the observed spectrum-weighted point estimate
    while resampling whole clusters. The equal-cluster estimator gives every
    unique modified sequence+charge cluster equal weight.
    """
    valid = frame[[cluster_column, delta_column]].copy()
    valid[delta_column] = pd.to_numeric(valid[delta_column], errors="coerce")
    valid = valid[np.isfinite(valid[delta_column])]
    if valid.empty:
        raise RuntimeError(f"No finite values for {delta_column}")

    grouped = valid.groupby(cluster_column, sort=True)[delta_column].agg(["sum", "count", "mean"])
    cluster_sums = grouped["sum"].to_numpy(dtype=np.float64)
    cluster_counts = grouped["count"].to_numpy(dtype=np.float64)
    cluster_means = grouped["mean"].to_numpy(dtype=np.float64)
    n_clusters = len(grouped)

    rng = np.random.default_rng(seed)
    spectrum_weighted = np.empty(reps, dtype=np.float64)
    cluster_equal = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, block_size):
        current = min(block_size, reps - start)
        indices = rng.integers(0, n_clusters, size=(current, n_clusters))
        sampled_sums = cluster_sums[indices].sum(axis=1)
        sampled_counts = cluster_counts[indices].sum(axis=1)
        spectrum_weighted[start : start + current] = sampled_sums / sampled_counts
        cluster_equal[start : start + current] = cluster_means[indices].mean(axis=1)

    spectrum_ci = np.quantile(spectrum_weighted * scale, [0.025, 0.975])
    cluster_ci = np.quantile(cluster_equal * scale, [0.025, 0.975])
    return {
        "cluster_definition": "modified sequence + modification sites + precursor charge",
        "n_spectra": int(len(valid)),
        "n_clusters": int(n_clusters),
        "bootstrap_replicates": int(reps),
        "spectrum_weighted_ratio": {
            "point_estimate": float(valid[delta_column].mean()) * scale,
            "ci95_percentile": [float(spectrum_ci[0]), float(spectrum_ci[1])],
        },
        "cluster_equal_weight": {
            "point_estimate": float(cluster_means.mean()) * scale,
            "ci95_percentile": [float(cluster_ci[0]), float(cluster_ci[1])],
        },
    }

