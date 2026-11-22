
#!/usr/bin/env python
"""
Experiment 22.24C — Bootstrap Bias Decomposition Audit.

Diagnostic only:
- Reuses frozen 22.24B event rankings.
- Reconstructs event-level exposure/opportunity using the
  existing 22.24C contribution builder.
- Measures conversation and category-cell influence.
- Reproduces the original conversation bootstrap.
- Separates baseline and proposed bootstrap shifts.
- Compares event-weighted and equal-conversation-weighted
  sensitivity estimands.

Does NOT:
- Train KGSF.
- Run retrieval.
- Change rankings.
- Modify the original 22.24B/22.24C outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
SCRIPT_22_24C = ROOT / "22_24C_independent_test_statistical_inference.py"

RESULTS_DIR = ROOT / "results_final_test_m2/experiment_22_24B"
INFERENCE_DIR = ROOT / "results_final_test_m2/experiment_22_24C"

KBRD_FILE = (
    ROOT / "results_final_test_m2/experiment_22_22/"
    "kbrd_independent_test_top500.csv"
)

CATALOG_FILE = ROOT / "catalog_audit_corrected.pkl"

CATEGORY_PROFILES = (
    ROOT / "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

REFERENCE_FILE = ROOT / "22_16_category_context_utility_calibration.py"

OUTPUT_DIR = INFERENCE_DIR / "bootstrap_bias_decomposition"

N_BOOT = 10000
SEED = 20261115
MIN_CELL_CONVERSATIONS = 20
EPS = 1e-12
BATCH_SIZE = 250


# ============================================================
# MODULE LOADING
# ============================================================

def load_module(path, module_name):

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name, path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def json_safe(value):

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# FAIRNESS METRIC
# ============================================================

def fairness_mad(exposure, opportunity, mask):

    exposure = np.asarray(exposure, dtype=np.float64)
    opportunity = np.asarray(opportunity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if np.any(opportunity[mask] <= EPS):
        raise ValueError(
            "A selected fairness cell has zero opportunity."
        )

    rates = (
        (exposure[mask] + EPS)
        / (opportunity[mask] + EPS)
    )

    log_rates = np.log(rates)

    return float(
        np.mean(
            np.abs(log_rates - np.mean(log_rates))
        )
    )


# ============================================================
# CLUSTER AGGREGATION
# ============================================================

def aggregate_clusters(values, event_cluster_index, n_clusters):

    result = np.zeros(
        (n_clusters,) + values.shape[1:],
        dtype=np.float64
    )

    np.add.at(result, event_cluster_index, values)

    return result


# ============================================================
# CONVERSATION INFLUENCE
# ============================================================

def conversation_influence(
    cluster_baseline,
    cluster_proposed,
    cluster_opportunity,
    cluster_count,
    conversation_ids,
    cell_mask,
    observed_baseline,
    observed_proposed
):

    full_delta = observed_proposed - observed_baseline

    total_baseline = cluster_baseline.sum(axis=0)
    total_proposed = cluster_proposed.sum(axis=0)
    total_opportunity = cluster_opportunity.sum(axis=0)

    rows = []

    for i, conv_id in enumerate(conversation_ids):

        b_exp = total_baseline - cluster_baseline[i]
        p_exp = total_proposed - cluster_proposed[i]
        opp = total_opportunity - cluster_opportunity[i]

        # Recalculate the exact statistic after deleting
        # one conversation from the aggregate.

        try:

            b_mad = fairness_mad(
                b_exp, opp, cell_mask
            )

            p_mad = fairness_mad(
                p_exp, opp, cell_mask
            )

            delta = p_mad - b_mad

        except ValueError:

            b_mad = np.nan
            p_mad = np.nan
            delta = np.nan

        rows.append({

            "conversation_id": int(conv_id),

            "events": int(cluster_count[i]),

            "event_share_percent": (
                100.0 * cluster_count[i]
                / cluster_count.sum()
            ),

            "baseline_exposure_total": float(
                cluster_baseline[i].sum()
            ),

            "proposed_exposure_total": float(
                cluster_proposed[i].sum()
            ),

            "opportunity_total": float(
                cluster_opportunity[i].sum()
            ),

            "baseline_MAD_without_conversation": b_mad,

            "proposed_MAD_without_conversation": p_mad,

            "delta_MAD_without_conversation": delta,

            "change_in_delta_after_deletion": (
                delta - full_delta
                if np.isfinite(delta)
                else np.nan
            ),

            "absolute_influence": (
                abs(delta - full_delta)
                if np.isfinite(delta)
                else np.nan
            ),

        })

    result = pd.DataFrame(rows)

    return result.sort_values(
        "absolute_influence",
        ascending=False
    )


# ============================================================
# CATEGORY-GROUP CELL INFLUENCE
# ============================================================

def cell_influence(
    cells,
    baseline_exposure,
    proposed_exposure,
    opportunity,
    cell_mask
):

    b_rates = (
        baseline_exposure + EPS
    ) / (opportunity + EPS)

    p_rates = (
        proposed_exposure + EPS
    ) / (opportunity + EPS)

    b_logs = np.log(b_rates[cell_mask])
    p_logs = np.log(p_rates[cell_mask])

    b_center = float(np.mean(b_logs))
    p_center = float(np.mean(p_logs))

    selected_indices = np.flatnonzero(cell_mask)

    full_b = fairness_mad(
        baseline_exposure, opportunity, cell_mask
    )

    full_p = fairness_mad(
        proposed_exposure, opportunity, cell_mask
    )

    full_delta = full_p - full_b

    rows = []

    for j in selected_indices:

        reduced_mask = cell_mask.copy()
        reduced_mask[j] = False

        reduced_b = fairness_mad(
            baseline_exposure,
            opportunity,
            reduced_mask
        )

        reduced_p = fairness_mad(
            proposed_exposure,
            opportunity,
            reduced_mask
        )

        reduced_delta = reduced_p - reduced_b

        rows.append({

            "category": str(cells[j][0]),

            "popularity_group": int(cells[j][1]),

            "baseline_exposure": float(
                baseline_exposure[j]
            ),

            "proposed_exposure": float(
                proposed_exposure[j]
            ),

            "opportunity": float(
                opportunity[j]
            ),

            "baseline_log_exposure_opportunity": float(
                b_logs[
                    np.where(selected_indices == j)[0][0]
                ]
            ),

            "proposed_log_exposure_opportunity": float(
                p_logs[
                    np.where(selected_indices == j)[0][0]
                ]
            ),

            "baseline_absolute_deviation": float(
                abs(b_logs[
                    np.where(selected_indices == j)[0][0]
                ] - b_center)
            ),

            "proposed_absolute_deviation": float(
                abs(p_logs[
                    np.where(selected_indices == j)[0][0]
                ] - p_center)
            ),

            "baseline_MAD_without_cell": reduced_b,

            "proposed_MAD_without_cell": reduced_p,

            "delta_MAD_without_cell": reduced_delta,

            "change_in_delta_after_cell_deletion": (
                reduced_delta - full_delta
            ),

        })

    return pd.DataFrame(rows).sort_values(
        "change_in_delta_after_cell_deletion",
        key=lambda x: x.abs(),
        ascending=False
    )


# ============================================================
# BOOTSTRAP DECOMPOSITION
# ============================================================

def run_bootstrap_decomposition(
    cluster_baseline,
    cluster_proposed,
    cluster_opportunity,
    cluster_count,
    cell_mask,
    observed_baseline,
    observed_proposed
):

    n_clusters = len(cluster_count)

    rng = np.random.default_rng(SEED)

    baseline_mad = np.empty(N_BOOT)
    proposed_mad = np.empty(N_BOOT)
    delta_mad = np.empty(N_BOOT)

    # Alternative equal-conversation sensitivity estimand.
    # Each conversation contributes its own event-average
    # exposure and opportunity instead of its event totals.

    normalized_baseline = (
        cluster_baseline
        / cluster_count[:, None]
    )

    normalized_proposed = (
        cluster_proposed
        / cluster_count[:, None]
    )

    normalized_opportunity = (
        cluster_opportunity
        / cluster_count[:, None]
    )

    equal_baseline_mad = np.empty(N_BOOT)
    equal_proposed_mad = np.empty(N_BOOT)
    equal_delta_mad = np.empty(N_BOOT)

    for start in range(0, N_BOOT, BATCH_SIZE):

        stop = min(start + BATCH_SIZE, N_BOOT)

        batch_size = stop - start

        # Identical sampling structure to original 22.24C.
        sampled = rng.integers(
            0,
            n_clusters,
            size=(batch_size, n_clusters)
        )

        multiplicity = np.zeros(
            (batch_size, n_clusters),
            dtype=np.float64
        )

        rows = np.repeat(
            np.arange(batch_size),
            n_clusters
        )

        np.add.at(
            multiplicity,
            (rows, sampled.ravel()),
            1.0
        )

        # Original event-weighted estimand.
        boot_b = multiplicity @ cluster_baseline
        boot_p = multiplicity @ cluster_proposed
        boot_o = multiplicity @ cluster_opportunity

        # Equal-conversation sensitivity estimand.
        eq_b = multiplicity @ normalized_baseline
        eq_p = multiplicity @ normalized_proposed
        eq_o = multiplicity @ normalized_opportunity

        for j in range(batch_size):

            idx = start + j

            baseline_mad[idx] = fairness_mad(
                boot_b[j],
                boot_o[j],
                cell_mask
            )

            proposed_mad[idx] = fairness_mad(
                boot_p[j],
                boot_o[j],
                cell_mask
            )

            delta_mad[idx] = (
                proposed_mad[idx]
                - baseline_mad[idx]
            )

            equal_baseline_mad[idx] = fairness_mad(
                eq_b[j],
                eq_o[j],
                cell_mask
            )

            equal_proposed_mad[idx] = fairness_mad(
                eq_p[j],
                eq_o[j],
                cell_mask
            )

            equal_delta_mad[idx] = (
                equal_proposed_mad[idx]
                - equal_baseline_mad[idx]
            )

    return {

        "baseline": baseline_mad,

        "proposed": proposed_mad,

        "delta": delta_mad,

        "equal_baseline": equal_baseline_mad,

        "equal_proposed": equal_proposed_mad,

        "equal_delta": equal_delta_mad,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=N_BOOT
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR
    )

    args = parser.parse_args()

    if args.bootstrap_replicates != N_BOOT:

        raise ValueError(
            "Use the original 10,000 replicates for exact "
            "reproduction of Experiment 22.24C."
        )

    print("=" * 78)
    print("EXPERIMENT 22.24C — BOOTSTRAP BIAS DECOMPOSITION")
    print("=" * 78)

    # Load the existing 22.24C implementation.
    exp = load_module(
        SCRIPT_22_24C,
        "experiment_22_24c_reference"
    )

    ref = exp.load_reference(REFERENCE_FILE)

    # Load frozen inputs.
    baseline = pd.read_csv(
        RESULTS_DIR / "baseline_event_results.csv"
    )

    proposed = pd.read_csv(
        RESULTS_DIR / "proposed_event_results.csv"
    )

    kbrd = pd.read_csv(KBRD_FILE)

    baseline = baseline.sort_values(
        "event_id"
    ).reset_index(drop=True)

    proposed = proposed.sort_values(
        "event_id"
    ).reset_index(drop=True)

    kbrd = kbrd.sort_values(
        "event_id"
    ).reset_index(drop=True)

    if not (
        baseline.event_id.equals(proposed.event_id)
        and baseline.event_id.equals(kbrd.event_id)
    ):

        raise RuntimeError(
            "Frozen event alignment failed."
        )

    if len(baseline) != 4003:

        raise RuntimeError(
            f"Expected 4003 events, found {len(baseline)}."
        )

    if baseline.conv_id.nunique() != 1283:

        raise RuntimeError(
            "Expected exactly 1283 conversations."
        )

    item2group, catalog, categories = ref.load_metadata(
        CATALOG_FILE,
        CATEGORY_PROFILES
    )

    # Reconstruct the same fixed category-group universe.
    all_cells = set()

    for value in kbrd.top_movie_ids:

        for movie_id in exp.parse_ids(value):

            movie_id = int(movie_id)

            group = int(item2group[movie_id])

            for category in categories.get(movie_id, []):

                all_cells.add(
                    (category, group)
                )

    cells = sorted(all_cells)

    # Reconstruct exact event-level sufficient statistics.
    b_ids, convs, bh, bn, be, bo = (
        exp.build_event_contributions(
            kbrd,
            baseline,
            item2group,
            categories,
            cells,
            ref
        )
    )

    p_ids, p_convs, ph, pn, pe, po = (
        exp.build_event_contributions(
            kbrd,
            proposed,
            item2group,
            categories,
            cells,
            ref
        )
    )

    if b_ids != p_ids or not np.array_equal(convs, p_convs):

        raise RuntimeError(
            "Baseline/proposed event alignment failed."
        )

    if not np.allclose(bo, po):

        raise RuntimeError(
            "Baseline/proposed opportunity contributions differ."
        )

    # Fixed support mask from original 22.24C.
    support = pd.read_csv(
        INFERENCE_DIR / "fairness_cell_support.csv"
    )

    support_map = {
        (str(row.category), int(row.popularity_group)):
            bool(row.included_for_inference)

        for row in support.itertuples()
    }

    cell_mask = np.asarray([
        support_map[(str(category), int(group))]
        for category, group in cells
    ], dtype=bool)

    if int(cell_mask.sum()) != 106:

        raise RuntimeError(
            "The reconstructed fairness universe does not "
            "contain the expected 106 cells."
        )

    # Conversation indexing.
    conversation_ids = np.unique(convs)

    conv_to_idx = {
        int(c): i
        for i, c in enumerate(conversation_ids)
    }

    event_cluster_index = np.asarray([
        conv_to_idx[int(c)]
        for c in convs
    ], dtype=np.int64)

    n_clusters = len(conversation_ids)

    cluster_count = np.bincount(
        event_cluster_index,
        minlength=n_clusters
    ).astype(np.float64)

    cluster_baseline = aggregate_clusters(
        be,
        event_cluster_index,
        n_clusters
    )

    cluster_proposed = aggregate_clusters(
        pe,
        event_cluster_index,
        n_clusters
    )

    cluster_opportunity = aggregate_clusters(
        bo,
        event_cluster_index,
        n_clusters
    )

    # Original observed point estimates.
    total_opportunity = bo.sum(axis=0)

    total_baseline = be.sum(axis=0)

    total_proposed = pe.sum(axis=0)

    observed_baseline = fairness_mad(
        total_baseline,
        total_opportunity,
        cell_mask
    )

    observed_proposed = fairness_mad(
        total_proposed,
        total_opportunity,
        cell_mask
    )

    observed_delta = (
        observed_proposed - observed_baseline
    )

    print("\n========== OBSERVED FAIRNESS ==========")

    print("Baseline MAD:", observed_baseline)
    print("Proposed MAD:", observed_proposed)
    print("Observed delta:", observed_delta)

    # --------------------------------------------------------
    # 1. CONVERSATION INFLUENCE
    # --------------------------------------------------------

    print("\n========== CONVERSATION INFLUENCE ==========")

    conversation_df = conversation_influence(
        cluster_baseline,
        cluster_proposed,
        cluster_opportunity,
        cluster_count,
        conversation_ids,
        cell_mask,
        observed_baseline,
        observed_proposed
    )

    # --------------------------------------------------------
    # 2. CATEGORY CELL INFLUENCE
    # --------------------------------------------------------

    print("\n========== CATEGORY CELL INFLUENCE ==========")

    cell_df = cell_influence(
        cells,
        total_baseline,
        total_proposed,
        total_opportunity,
        cell_mask
    )

    # --------------------------------------------------------
    # 3. BOOTSTRAP BIAS DECOMPOSITION
    # --------------------------------------------------------

    print("\n========== REPRODUCING ORIGINAL BOOTSTRAP ==========")

    bootstrap = run_bootstrap_decomposition(
        cluster_baseline,
        cluster_proposed,
        cluster_opportunity,
        cluster_count,
        cell_mask,
        observed_baseline,
        observed_proposed
    )

    archive_path = (
        INFERENCE_DIR / "bootstrap_distributions.npz"
    )

    with np.load(
        archive_path,
        allow_pickle=False
    ) as archive:

        original_delta = archive["delta_fairness_mad"]

    exact_reproduction = bool(
        np.array_equal(
            original_delta,
            bootstrap["delta"]
        )
    )

    max_difference = float(
        np.max(
            np.abs(original_delta - bootstrap["delta"])
        )
    )

    print("Original bootstrap exact match:", exact_reproduction)
    print("Maximum absolute difference:", max_difference)

    if not np.allclose(
        original_delta,
        bootstrap["delta"],
        rtol=0,
        atol=1e-12
    ):

        raise RuntimeError(
            "Bootstrap reproduction failed. "
            "Do not interpret the decomposition."
        )

    # --------------------------------------------------------
    # 4. DECOMPOSE BASELINE AND PROPOSED SHIFTS
    # --------------------------------------------------------

    bootstrap_baseline_mean = float(
        np.mean(bootstrap["baseline"])
    )

    bootstrap_proposed_mean = float(
        np.mean(bootstrap["proposed"])
    )

    baseline_shift = (
        bootstrap_baseline_mean - observed_baseline
    )

    proposed_shift = (
        bootstrap_proposed_mean - observed_proposed
    )

    delta_shift = (
        np.mean(bootstrap["delta"]) - observed_delta
    )

    # Algebraic identity:
    # Delta bootstrap bias = proposed shift - baseline shift.
    decomposition_identity = bool(
        np.isclose(
            delta_shift,
            proposed_shift - baseline_shift,
            atol=1e-10
        )
    )

    # --------------------------------------------------------
    # 5. EQUAL-CONVERSATION WEIGHTING SENSITIVITY
    # --------------------------------------------------------

    normalized_baseline = (
    cluster_baseline / cluster_count[:, None]
    )

    normalized_proposed = (
    cluster_proposed / cluster_count[:, None]
    )

    normalized_opportunity = (
    cluster_opportunity / cluster_count[:, None]
    )

    equal_baseline_point = fairness_mad(
    normalized_baseline.sum(axis=0),
    normalized_opportunity.sum(axis=0),
    cell_mask
    )

    equal_proposed_point = fairness_mad(
    normalized_proposed.sum(axis=0),
    normalized_opportunity.sum(axis=0),
    cell_mask
    )

    # The common scalar division cancels in exposure/opportunity.
    # Therefore this point estimate is identical to the event-
    # weighted point estimate. The bootstrap sensitivity below
    # instead normalizes each conversation before aggregation.
    #
    # This is a distinct weighting estimand, not a replacement
    # for the original 22.24C estimand.

    equal_boot_delta = bootstrap["equal_delta"]

    rows = [

        {
            "analysis": "original_event_weighted",

            "baseline_observed": observed_baseline,

            "proposed_observed": observed_proposed,

            "observed_delta": observed_delta,

            "bootstrap_baseline_mean": bootstrap_baseline_mean,

            "bootstrap_proposed_mean": bootstrap_proposed_mean,

            "bootstrap_delta_mean": float(
                np.mean(bootstrap["delta"])
            ),

            "bootstrap_bias": delta_shift,

            "bootstrap_sd": float(
                np.std(bootstrap["delta"], ddof=1)
            ),

            "bootstrap_q025": float(
                np.quantile(bootstrap["delta"], 0.025)
            ),

            "bootstrap_q975": float(
                np.quantile(bootstrap["delta"], 0.975)
            ),

        },

        {
            "analysis": "equal_conversation_normalized_sensitivity",

            "baseline_observed": equal_baseline_point,

            "proposed_observed": equal_proposed_point,

            "observed_delta": (
                equal_proposed_point - equal_baseline_point
            ),

            "bootstrap_baseline_mean": float(
                np.mean(bootstrap["equal_baseline"])
            ),

            "bootstrap_proposed_mean": float(
                np.mean(bootstrap["equal_proposed"])
            ),

            "bootstrap_delta_mean": float(
                np.mean(equal_boot_delta)
            ),

            "bootstrap_bias": float(
                np.mean(equal_boot_delta)
                - (equal_proposed_point - equal_baseline_point)
            ),

            "bootstrap_sd": float(
                np.std(equal_boot_delta, ddof=1)
            ),

            "bootstrap_q025": float(
                np.quantile(equal_boot_delta, 0.025)
            ),

            "bootstrap_q975": float(
                np.quantile(equal_boot_delta, 0.975)
            ),

        }

    ]

    decomposition_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 6. DIAGNOSTICS
    # --------------------------------------------------------

    finite_influence = conversation_df[
        "absolute_influence"
    ].dropna()

    top_1_percent = max(
        1,
        int(np.ceil(0.01 * len(finite_influence)))
    )

    top_5_percent = max(
        1,
        int(np.ceil(0.05 * len(finite_influence)))
    )

    influence_sorted = finite_influence.sort_values(
        ascending=False
    )

    top_1_influence = float(
        influence_sorted.head(top_1_percent).sum()
    )

    total_absolute_influence = float(
        influence_sorted.sum()
    )

    concentration_1 = (
        top_1_influence / total_absolute_influence
        if total_absolute_influence > 0
        else 0.0
    )

    concentration_5 = (
        float(influence_sorted.head(top_5_percent).sum())
        / total_absolute_influence
        if total_absolute_influence > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 7. SAVE OUTPUTS
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    conversation_df.to_csv(
        args.output / "conversation_influence.csv",
        index=False
    )

    cell_df.to_csv(
        args.output / "cell_influence.csv",
        index=False
    )

    decomposition_df.to_csv(
        args.output / "bootstrap_bias_decomposition.csv",
        index=False
    )

    diagnostics = {

        "experiment": "22.24C_bootstrap_bias_decomposition",

        "diagnostic_only": True,

        "events": int(len(convs)),

        "conversations": int(n_clusters),

        "fairness_cells_total": int(len(cells)),

        "fairness_cells_included": int(cell_mask.sum()),

        "bootstrap_replicates": N_BOOT,

        "bootstrap_seed": SEED,

        "bootstrap_exact_reproduction": exact_reproduction,

        "bootstrap_max_absolute_difference": max_difference,

        "observed": {

            "baseline_MAD": observed_baseline,

            "proposed_MAD": observed_proposed,

            "delta_MAD": observed_delta,

        },

        "bootstrap": {

            "baseline_mean": bootstrap_baseline_mean,

            "proposed_mean": bootstrap_proposed_mean,

            "delta_mean": float(
                np.mean(bootstrap["delta"])
            ),

            "baseline_shift": baseline_shift,

            "proposed_shift": proposed_shift,

            "delta_shift": delta_shift,

            "decomposition_identity_pass": decomposition_identity,

            "delta_standard_deviation": float(
                np.std(bootstrap["delta"], ddof=1)
            ),

            "delta_q025": float(
                np.quantile(bootstrap["delta"], 0.025)
            ),

            "delta_median": float(
                np.median(bootstrap["delta"])
            ),

            "delta_q975": float(
                np.quantile(bootstrap["delta"], 0.975)
            ),

        },

        "conversation_influence": {

            "maximum_absolute_influence": float(
                finite_influence.max()
            ),

            "median_absolute_influence": float(
                finite_influence.median()
            ),

            "top_1_percent_share_of_absolute_influence": concentration_1,

            "top_5_percent_share_of_absolute_influence": concentration_5,

            "largest_influence_conversation": int(
                conversation_df.iloc[0]["conversation_id"]
            ),

        },

        "alternative_weighting": {

            "equal_conversation_weighted_sensitivity": True,

            "is_primary_estimand": False,

            "bootstrap_mean_delta": float(
                np.mean(equal_boot_delta)
            ),

            "bootstrap_sd": float(
                np.std(equal_boot_delta, ddof=1)
            ),

        },

        "limitations": [

            "This audit conditions on frozen sequential rankings.",

            "Conversation deletion measures influence on the aggregate statistic, not causal effects.",

            "Equal-conversation normalization defines a different sensitivity estimand.",

            "This diagnostic does not establish policy-level statistical significance.",

            "The global sequential exposure ledger remains a source of cross-conversation dependence."

        ],

        "preservation": {

            "training_performed": False,

            "retrieval_performed": False,

            "ranking_policy_modified": False,

            "independent_test_rerun": False,

            "original_22_24B_outputs_modified": False,

            "original_22_24C_outputs_modified": False,

        }

    }

    diagnostics = json_safe(diagnostics)

    (args.output / "bias_diagnostics.json").write_text(
        json.dumps(
            diagnostics,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 8. HUMAN-READABLE REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "EXPERIMENT 22.24C — BOOTSTRAP BIAS DECOMPOSITION REPORT"
    )

    report.append("=" * 72)

    report.append("")

    report.append("OBSERVED FAIRNESS EFFECT")

    report.append(
        f"Baseline MAD: {observed_baseline:.8f}"
    )

    report.append(
        f"Proposed MAD: {observed_proposed:.8f}"
    )

    report.append(
        f"Observed delta: {observed_delta:.8f}"
    )

    report.append("")

    report.append("BOOTSTRAP DECOMPOSITION")

    report.append(
        f"Baseline bootstrap mean: {bootstrap_baseline_mean:.8f}"
    )

    report.append(
        f"Proposed bootstrap mean: {bootstrap_proposed_mean:.8f}"
    )

    report.append(
        f"Baseline shift: {baseline_shift:.8f}"
    )

    report.append(
        f"Proposed shift: {proposed_shift:.8f}"
    )

    report.append(
        f"Difference in shifts: {delta_shift:.8f}"
    )

    report.append(
        f"Decomposition identity passed: {decomposition_identity}"
    )

    report.append("")

    report.append("CONVERSATION INFLUENCE")

    report.append(
        f"Top 1% influence concentration: {concentration_1:.4%}"
    )

    report.append(
        f"Top 5% influence concentration: {concentration_5:.4%}"
    )

    report.append(
        f"Largest influence conversation: "
        f"{int(conversation_df.iloc[0]['conversation_id'])}"
    )

    report.append("")

    report.append("BOOTSTRAP REPRODUCTION")

    report.append(
        f"Exact original bootstrap reproduction: {exact_reproduction}"
    )

    report.append(
        f"Maximum absolute difference: {max_difference:.12g}"
    )

    report.append("")

    report.append("ALTERNATIVE WEIGHTING")

    report.append(
        "Equal-conversation-normalized sensitivity delta mean: "
        f"{np.mean(equal_boot_delta):.8f}"
    )

    report.append("")

    report.append("INTERPRETATION")

    report.append(
        "This diagnostic investigates the numerical decomposition "
        "of the fairness bootstrap distribution. It does not establish "
        "confirmatory policy-level statistical significance."
    )

    report.append(
        "The original event-weighted estimand remains the primary "
        "22.24C fairness estimand."
    )

    report.append(
        "The equal-conversation-normalized result is a separate "
        "sensitivity estimand."
    )

    report.append("")

    report.append(
        "No model training, retrieval, ranking changes, or independent "
        "test reruns were performed."
    )

    report_text = "\n".join(report)

    (args.output / "bias_decomposition_report.txt").write_text(
        report_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n========== BOOTSTRAP DECOMPOSITION ==========")

    print(decomposition_df.to_string(index=False))

    print("\n========== CONVERSATION INFLUENCE ==========")

    print(
        conversation_df.head(10).to_string(index=False)
    )

    print("\n========== CELL INFLUENCE ==========")

    print(
        cell_df.head(10).to_string(index=False)
    )

    print("\n========== FINAL DIAGNOSTICS ==========")

    print(json.dumps(
        diagnostics,
        indent=2,
        allow_nan=False
    ))

    print("\nCreated:")

    for name in (
        "conversation_influence.csv",
        "cell_influence.csv",
        "bootstrap_bias_decomposition.csv",
        "bias_diagnostics.json",
        "bias_decomposition_report.txt"
    ):

        print(" ", args.output / name)

    print("\nDiagnostic complete.")

    print(
        "Original rankings, scripts and statistical outputs "
        "were not modified."
    )


if __name__ == "__main__":
    main()
