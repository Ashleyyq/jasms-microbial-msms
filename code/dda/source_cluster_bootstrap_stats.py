#!/usr/bin/env python3
"""Source-stratified modified-precursor bootstrap used by the postprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd


def source_stratified_cluster_bootstrap(
    frame: pd.DataFrame,
    delta_column: str,
    source_column: str,
    cluster_column: str,
    reps: int,
    seed: int,
    block_size: int = 32,
    scale: float = 1.0,
) -> dict:
    valid = frame[[source_column, cluster_column, delta_column]].copy()
    valid[delta_column] = pd.to_numeric(valid[delta_column], errors="coerce")
    valid = valid[np.isfinite(valid[delta_column])]
    if valid.empty:
        raise RuntimeError(f"No finite values for {delta_column}")

    strata = []
    source_points = []
    for source, source_frame in valid.groupby(source_column, sort=True):
        grouped = source_frame.groupby(cluster_column, sort=True)[delta_column].agg(
            ["sum", "count", "mean"]
        )
        strata.append(
            {
                "source": str(source),
                "sum": grouped["sum"].to_numpy(dtype=np.float64),
                "count": grouped["count"].to_numpy(dtype=np.float64),
                "mean": grouped["mean"].to_numpy(dtype=np.float64),
            }
        )
        source_points.append(float(grouped["mean"].mean()))

    total_clusters = sum(len(item["mean"]) for item in strata)
    rng = np.random.default_rng(seed)
    spectrum_weighted = np.empty(reps, dtype=np.float64)
    cluster_equal = np.empty(reps, dtype=np.float64)
    source_equal_macro = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, block_size):
        current = min(block_size, reps - start)
        sampled_sum = np.zeros(current, dtype=np.float64)
        sampled_count = np.zeros(current, dtype=np.float64)
        sampled_cluster_means = np.zeros(current, dtype=np.float64)
        sampled_source_macro = np.zeros(current, dtype=np.float64)
        for item in strata:
            n_clusters = len(item["mean"])
            indices = rng.integers(0, n_clusters, size=(current, n_clusters))
            sampled_sum += item["sum"][indices].sum(axis=1)
            sampled_count += item["count"][indices].sum(axis=1)
            selected_means = item["mean"][indices]
            sampled_cluster_means += selected_means.sum(axis=1)
            sampled_source_macro += selected_means.mean(axis=1)
        spectrum_weighted[start : start + current] = sampled_sum / sampled_count
        cluster_equal[start : start + current] = sampled_cluster_means / total_clusters
        source_equal_macro[start : start + current] = sampled_source_macro / len(strata)

    def estimate(point: float, samples: np.ndarray) -> dict:
        interval = np.quantile(samples * scale, [0.025, 0.975])
        return {
            "point_estimate": float(point) * scale,
            "ci95_percentile": [float(interval[0]), float(interval[1])],
        }

    source_cluster_means = (
        valid.groupby([source_column, cluster_column], sort=True)[delta_column].mean()
    )
    return {
        "sampling_unit": "source-specific modified sequence + modification sites + precursor charge",
        "stratification": "source; fixed observed number of clusters per source",
        "n_spectra": int(len(valid)),
        "n_sources": int(len(strata)),
        "n_source_specific_clusters": int(total_clusters),
        "clusters_per_source": {
            item["source"]: int(len(item["mean"])) for item in strata
        },
        "bootstrap_replicates": int(reps),
        "spectrum_weighted_ratio": estimate(
            float(valid[delta_column].mean()), spectrum_weighted
        ),
        "source_stratified_cluster_equal_weight": estimate(
            float(source_cluster_means.mean()), cluster_equal
        ),
        "source_equal_cluster_macro": estimate(
            float(np.mean(source_points)), source_equal_macro
        ),
    }

