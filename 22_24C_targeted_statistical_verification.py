#!/usr/bin/env python
"""
Targeted statistical verification for Experiment 22.24C.

This is a diagnostic only. It does not train models, rerun retrieval, change
rankings, or regenerate the original bootstrap distribution.

It:
  1. Reconstructs the observed paired estimands from frozen 22.24B files.
  2. Loads the existing 10,000 conversation-cluster bootstrap replicates.
  3. Computes percentile, basic, and BCa intervals.
  4. Computes BCa acceleration from leave-one-conversation-out jackknife
     estimates using the same fixed fairness-cell universe.
  5. Reports bootstrap bias, skewness, tail support, and the existing
     null-centred bootstrap p-value approximation with Holm adjustment.

Run from the project root:
python .\\22_24C_targeted_statistical_verification.py

Optional:
python .\\22_24C_targeted_statistical_verification.py --replicates 10000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("22_24C_independent_test_statistical_inference.py")
DEFAULT_BOOTSTRAP = Path(
    "results_final_test_m2/experiment_22_24C/bootstrap_distributions.npz"
)
DEFAULT_OUT = Path(
    "results_final_test_m2/experiment_22_24C/targeted_verification"
)
NORMAL = NormalDist()
EPS = 1e-12


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_22_24c_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def holm_adjust(p_values):
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(p)))
        adjusted[name] = running
    return adjusted


def two_sided_centered_bootstrap_p(values, observed):
    """Reproduce the original script's approximate null-centred bootstrap test."""
    errors = np.asarray(values, dtype=float) - float(observed)
    extreme = np.count_nonzero(np.abs(errors) >= abs(float(observed)))
    return float((extreme + 1) / (len(errors) + 1))


def skewness(values):
    x = np.asarray(values, dtype=float)
    s = np.std(x, ddof=1)
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - np.mean(x)) / s) ** 3))


def bca_interval(boot, observed, jackknife, alpha=0.05):
    """
    Return BCa interval and diagnostics.
    If the empirical bias correction is at/near its finite-sample boundary,
    mark BCa unstable rather than presenting a misleading interval.
    """
    boot = np.asarray(boot, dtype=float)
    jack = np.asarray(jackknife, dtype=float)
    observed = float(observed)

    less = np.count_nonzero(boot < observed)
    equal = np.count_nonzero(boot == observed)
    prop_less = (less + 0.5 * equal) / len(boot)
    # Finite Monte-Carlo clipping avoids +/- infinity, while retaining a flag.
    clipped_prop = min(max(prop_less, 0.5 / len(boot)),
                       1.0 - 0.5 / len(boot))
    z0 = NORMAL.inv_cdf(clipped_prop)

    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    sum_sq = float(np.sum(diffs ** 2))
    sum_cube = float(np.sum(diffs ** 3))
    if sum_sq <= 0:
        acceleration = 0.0
    else:
        acceleration = sum_cube / (6.0 * (sum_sq ** 1.5))

    adjusted_probs = []
    for nominal in (alpha / 2.0, 1.0 - alpha / 2.0):
        z_alpha = NORMAL.inv_cdf(nominal)
        denom = 1.0 - acceleration * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            return {
                "low": None, "high": None, "z0": z0,
                "acceleration": acceleration,
                "bias_correction_proportion": prop_less,
                "adjusted_probability_low": None,
                "adjusted_probability_high": None,
                "stable": False,
                "reason": "BCa adjustment denominator is numerically singular.",
            }
        adjusted_z = z0 + (z0 + z_alpha) / denom
        adjusted_p = NORMAL.cdf(adjusted_z)
        adjusted_probs.append(adjusted_p)

    p_low, p_high = adjusted_probs
    stable = (
        np.isfinite(p_low) and np.isfinite(p_high)
        and 0.0 < p_low < p_high < 1.0
        and p_low >= 1.0 / (len(boot) * 2.0)
        and p_high <= 1.0 - 1.0 / (len(boot) * 2.0)
    )
    if not stable:
        return {
            "low": None, "high": None, "z0": z0,
            "acceleration": acceleration,
            "bias_correction_proportion": prop_less,
            "adjusted_probability_low": p_low,
            "adjusted_probability_high": p_high,
            "stable": False,
            "reason": (
                "BCa adjusted tail probability is outside reliable empirical "
                "bootstrap resolution; do not interpret this interval."
            ),
        }

    return {
        "low": float(np.quantile(boot, p_low)),
        "high": float(np.quantile(boot, p_high)),
        "z0": float(z0),
        "acceleration": float(acceleration),
        "bias_correction_proportion": float(prop_less),
        "adjusted_probability_low": float(p_low),
        "adjusted_probability_high": float(p_high),
        "stable": True,
        "reason": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args()

    for path in (args.source, args.bootstrap):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = load_module(args.source)
    ref = source.load_reference(source.DEFAULT_REFERENCE)
    item2group, _catalog, categories = ref.load_metadata(
        source.DEFAULT_CATALOG, source.DEFAULT_PROFILES
    )

    baseline = pd.read_csv(source.DEFAULT_DIR / "baseline_event_results.csv")
    proposed = pd.read_csv(source.DEFAULT_DIR / "proposed_event_results.csv")
    kbrd = pd.read_csv(source.DEFAULT_KBRD)

    baseline = baseline.sort_values("event_id").reset_index(drop=True)
    proposed = proposed.sort_values("event_id").reset_index(drop=True)
    kbrd = kbrd.sort_values("event_id").reset_index(drop=True)

    if not (
        baseline.event_id.equals(proposed.event_id)
        and baseline.event_id.equals(kbrd.event_id)
    ):
        raise RuntimeError("Baseline, proposed, and KBRD event IDs do not align.")
    if len(baseline) != 4003 or baseline.conv_id.nunique() != 1283:
        raise RuntimeError("Expected 4,003 events and 1,283 conversations.")

    all_cells = set()
    for value in kbrd.top_movie_ids:
        for movie_id in source.parse_ids(value):
            movie_id = int(movie_id)
            group = int(item2group[movie_id])
            for category in categories.get(movie_id, []):
                all_cells.add((category, group))
    cells = sorted(all_cells)

    b_ids, convs, bh, bn, be, bo = source.build_event_contributions(
        kbrd, baseline, item2group, categories, cells, ref
    )
    p_ids, p_convs, ph, pn, pe, po = source.build_event_contributions(
        kbrd, proposed, item2group, categories, cells, ref
    )
    if b_ids != p_ids or not np.array_equal(convs, p_convs):
        raise RuntimeError("Paired event/conversation alignment failed.")
    if not np.allclose(bo, po):
        raise RuntimeError("Baseline and proposed opportunity contributions differ.")

    unique_convs = np.unique(convs)
    conv_to_idx = {int(c): i for i, c in enumerate(unique_convs)}
    event_conv_idx = np.asarray([conv_to_idx[int(c)] for c in convs], dtype=int)
    n_conv = len(unique_convs)

    def cluster_sum(values):
        result = np.zeros((n_conv,) + values.shape[1:], dtype=np.float64)
        np.add.at(result, event_conv_idx, values)
        return result

    counts = np.bincount(event_conv_idx, minlength=n_conv).astype(float)
    cbh = cluster_sum(bh[:, None])[:, 0]
    cph = cluster_sum(ph[:, None])[:, 0]
    cbn = cluster_sum(bn[:, None])[:, 0]
    cpn = cluster_sum(pn[:, None])[:, 0]
    cbe, cpe = cluster_sum(be), cluster_sum(pe)
    copp = cluster_sum(bo)

    support = (copp > EPS).sum(axis=0)
    cell_mask = support >= source.MIN_CELL_CONVERSATIONS
    if int(cell_mask.sum()) != 106:
        raise RuntimeError(
            f"Expected 106 supported fairness cells, found {int(cell_mask.sum())}. "
            "Check that the frozen input files match the original run."
        )

    base_mad = source.mad_from_totals(
        cbe.sum(axis=0), copp.sum(axis=0), cell_mask
    )
    prop_mad = source.mad_from_totals(
        cpe.sum(axis=0), copp.sum(axis=0), cell_mask
    )
    observed = {
        "HR@10": float(ph.mean() - bh.mean()),
        "NDCG@10": float(pn.mean() - bn.mean()),
        "fairness_MAD": float(prop_mad - base_mad),
    }

    # Leave-one-conversation-out jackknife, with the same fixed cell mask.
    jack = {name: np.empty(n_conv, dtype=float) for name in observed}
    total_count = float(counts.sum())
    total_bh, total_ph = float(cbh.sum()), float(cph.sum())
    total_bn, total_pn = float(cbn.sum()), float(cpn.sum())
    total_be, total_pe, total_opp = cbe.sum(axis=0), cpe.sum(axis=0), copp.sum(axis=0)

    for i in range(n_conv):
        denom = total_count - counts[i]
        jack["HR@10"][i] = (
            (total_ph - cph[i]) - (total_bh - cbh[i])
        ) / denom
        jack["NDCG@10"][i] = (
            (total_pn - cpn[i]) - (total_bn - cbn[i])
        ) / denom
        base_i = source.mad_from_totals(
            total_be - cbe[i], total_opp - copp[i], cell_mask
        )
        prop_i = source.mad_from_totals(
            total_pe - cpe[i], total_opp - copp[i], cell_mask
        )
        jack["fairness_MAD"][i] = prop_i - base_i

    with np.load(args.bootstrap) as archive:
        boot = {
            "HR@10": np.asarray(archive["delta_hr10"], dtype=float),
            "NDCG@10": np.asarray(archive["delta_ndcg10"], dtype=float),
            "fairness_MAD": np.asarray(archive["delta_fairness_mad"], dtype=float),
        }

    if args.replicates is not None:
        boot = {k: v[:args.replicates] for k, v in boot.items()}
    for name, values in boot.items():
        if len(values) < 1000 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid bootstrap distribution for {name}.")

    p_raw = {
        name: two_sided_centered_bootstrap_p(boot[name], observed[name])
        for name in observed
    }
    p_holm = holm_adjust(p_raw)

    report = {}
    rows = []
    for name in observed:
        values = boot[name]
        theta = observed[name]
        q025, q975 = np.quantile(values, [0.025, 0.975])
        basic_low, basic_high = 2 * theta - q975, 2 * theta - q025
        bca = bca_interval(values, theta, jack[name])
        report[name] = {
            "observed_delta": theta,
            "bootstrap_replicates": int(len(values)),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "bootstrap_bias_mean_minus_observed": float(np.mean(values) - theta),
            "bootstrap_bias_percent_of_abs_observed": (
                float(100 * (np.mean(values) - theta) / abs(theta))
                if abs(theta) > 0 else None
            ),
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "bootstrap_skewness": skewness(values),
            "bootstrap_min": float(np.min(values)),
            "bootstrap_max": float(np.max(values)),
            "fraction_bootstrap_below_observed": float(np.mean(values < theta)),
            "fraction_bootstrap_equal_observed": float(np.mean(values == theta)),
            "percentile_ci95": [float(q025), float(q975)],
            "basic_ci95": [float(basic_low), float(basic_high)],
            "bca_ci95": bca,
            "jackknife_min": float(np.min(jack[name])),
            "jackknife_max": float(np.max(jack[name])),
            "jackknife_sd": float(np.std(jack[name], ddof=1)),
            "null_centered_bootstrap_p_approx": p_raw[name],
            "holm_adjusted_p_approx": p_holm[name],
        }
        rows.append({
            "metric": name,
            "observed_delta": theta,
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_bias": float(np.mean(values) - theta),
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "bootstrap_skewness": skewness(values),
            "percentile_ci95_low": float(q025),
            "percentile_ci95_high": float(q975),
            "basic_ci95_low": float(basic_low),
            "basic_ci95_high": float(basic_high),
            "bca_ci95_low": bca["low"],
            "bca_ci95_high": bca["high"],
            "bca_stable": bca["stable"],
            "bca_z0": bca["z0"],
            "bca_acceleration": bca["acceleration"],
            "null_centered_bootstrap_p_approx": p_raw[name],
            "holm_adjusted_p_approx": p_holm[name],
        })

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targeted_verification.json").write_text(
        json.dumps({
            "experiment": "22.24C_targeted_statistical_verification",
            "events": int(len(baseline)),
            "conversations": int(n_conv),
            "fairness_cells_total": int(len(cells)),
            "fairness_cells_included": int(cell_mask.sum()),
            "minimum_cell_conversations": int(source.MIN_CELL_CONVERSATIONS),
            "baseline_fairness_MAD": base_mad,
            "proposed_fairness_MAD": prop_mad,
            "metrics": report,
            "interpretation": (
                "BCa intervals are reported only when their adjusted tail "
                "probabilities are within the empirical resolution of the "
                "available bootstrap replicates. The null-centred p-values "
                "reproduce the original approximate procedure and should not "
                "be treated as an independent exact randomization test."
            ),
        }, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        args.out / "targeted_verification_summary.csv", index=False
    )

    print("=" * 78)
    print("EXPERIMENT 22.24C — TARGETED STATISTICAL VERIFICATION")
    print("=" * 78)
    print(f"Events: {len(baseline)} | Conversations: {n_conv}")
    print(f"Fairness cells: {int(cell_mask.sum())}/{len(cells)}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved:", args.out.resolve())


if __name__ == "__main__":
    main()
