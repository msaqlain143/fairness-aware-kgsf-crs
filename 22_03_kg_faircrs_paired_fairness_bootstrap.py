#!/usr/bin/env python
"""
Experiment 22.3: paired conversation-cluster bootstrap for KG-FairCRS fairness.

Proposal-aligned statistical evaluation:
- 10,000 paired conversation-cluster bootstrap replicates by default.
- Same sampled conversation multiplicities for every compared method.
- Duplicate sampled conversations retain multiplicity (all their events are repeated).
- Recomputes category-group log exposure/opportunity MAD per bootstrap sample.
- Computes paired HR@10 and NDCG@10 differences versus A1.
- Uses only the held-out validation partition already fixed in Experiments 22.0-22.2.
- No model training, candidate generation, parameter tuning, or test-set access.

Inputs:
  results_corrected/experiment_22_02/heldout_event_recommendations.csv
  frozen Experiment 22.1 inputs and category/catalog metadata.

The fairness statistic is the same category_group_log_rate_MAD used in 22.0-22.2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_A1 = "A1_frequency_3"
POLICY_METHODS = [
    "FULL_KG_FairCRS",
    "ABL_no_KG_support",
    "ABL_no_adaptive_calibration",
    "ABL_frozen_exposure_history",
]
SEED = 22300
EPS = 1e-12


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("kgfaircrs_exp22_1_for_22_3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import required helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_event_contribution(
    movie_ids, group_ids, item_categories, candidate_ids, item2group
):
    """Return category x popularity-group exposure and opportunity vectors."""
    categories = sorted({
        cat for cats in item_categories.values() for cat in cats
    })
    cat_index = {cat: i for i, cat in enumerate(categories)}
    n_cells = len(categories) * 5
    exposure = np.zeros(n_cells, dtype=np.float64)
    opportunity = np.zeros(n_cells, dtype=np.float64)

    for pos, (movie_id, group) in enumerate(zip(movie_ids, group_ids)):
        cats = item_categories.get(int(movie_id), [])
        if not cats:
            continue
        weight = 1.0 / np.log2(pos + 2.0)
        share = weight / len(cats)
        for cat in cats:
            exposure[cat_index[cat] * 5 + int(group)] += share

    n = len(candidate_ids)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.log2(ranks + 1.0)
    weights /= weights.sum() + EPS
    for j, movie_id in enumerate(candidate_ids):
        cats = item_categories.get(int(movie_id), [])
        if not cats:
            continue
        group = int(item2group[int(movie_id)])
        share = float(weights[j]) / len(cats)
        for cat in cats:
            opportunity[cat_index[cat] * 5 + group] += share

    return exposure, opportunity, categories


def fairness_from_totals(exposure, opportunity, n_categories):
    """Vectorized equivalent of category_disparity() for bootstrap replicates."""
    n_boot = exposure.shape[0]
    exp3 = exposure.reshape(n_boot, n_categories, 5)
    opp3 = opportunity.reshape(n_boot, n_categories, 5)

    rates = exp3 / (opp3 + EPS)
    valid = opp3 > EPS
    n_valid = valid.sum(axis=2)

    safe_logs = np.log(np.maximum(rates, EPS))
    log_sum = np.where(valid, safe_logs, 0.0).sum(axis=2)
    mean_log = log_sum / np.maximum(n_valid, 1)

    abs_dev = np.where(valid, np.abs(safe_logs - mean_log[:, :, None]), 0.0)
    category_mad = abs_dev.sum(axis=2) / np.maximum(n_valid, 1)

    eligible_categories = n_valid >= 2
    category_sum = np.where(eligible_categories, category_mad, 0.0).sum(axis=1)
    category_count = eligible_categories.sum(axis=1)
    result = category_sum / np.maximum(category_count, 1)
    result[category_count == 0] = np.nan
    return result


def ci_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": [None, None]}
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-script", type=Path,
        default=Path("22_01_kg_faircrs_relevance_gate_ablation.py"))
    parser.add_argument("--heldout-results", type=Path,
        default=Path("results_corrected/experiment_22_02/heldout_event_recommendations.csv"))
    parser.add_argument("--candidates", type=Path,
        default=Path("results_corrected/adaptive_m2_validation_export22_candidates.csv"))
    parser.add_argument("--events", type=Path,
        default=Path("results_corrected/adaptive_m2_validation_export22_events.csv"))
    parser.add_argument("--mapping", type=Path,
        default=Path("results_corrected/m2_event_conversation_mapping.csv"))
    parser.add_argument("--catalog-pickle", type=Path,
        default=Path("catalog_audit_corrected.pkl"))
    parser.add_argument("--category-profiles", type=Path,
        default=Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    parser.add_argument("--out", type=Path,
        default=Path("results_corrected/experiment_22_03"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.bootstrap < 1000:
        raise ValueError("Proposal-aligned analysis requires at least 1,000 replicates; default is 10,000.")

    M = load_module(args.base_script)

    class InputArgs:
        pass
    inp = InputArgs()
    inp.candidates = args.candidates
    inp.events = args.events
    inp.mapping = args.mapping
    inp.catalog_pickle = args.catalog_pickle
    inp.category_profiles = args.category_profiles

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 22.3 — PAIRED CONVERSATION-CLUSTER FAIRNESS INFERENCE")
    print("=" * 78)
    print("Loading validated metadata and frozen candidate pool...")
    data = M.load_inputs(inp)

    policy_df = pd.read_csv(args.heldout_results)
    required = {
        "event_id", "conv_id", "method", "hit10", "ndcg10",
        "top10_movie_ids", "top10_group_ids"
    }
    if not required.issubset(policy_df.columns):
        raise ValueError(f"22.2 held-out file missing columns: {required - set(policy_df.columns)}")

    event_ids = sorted(map(int, policy_df.event_id.unique()))
    conv_map = (
        policy_df[["event_id", "conv_id"]]
        .drop_duplicates()
        .set_index("event_id")["conv_id"]
        .astype(int).to_dict()
    )
    if len(event_ids) != 624:
        raise ValueError(f"Expected 624 held-out events from Experiment 22.2, found {len(event_ids)}.")
    if len(set(conv_map.values())) != 176:
        raise ValueError("Expected 176 held-out conversations.")

    for method in POLICY_METHODS:
        subset = policy_df[policy_df.method == method]
        if len(subset) != len(event_ids) or subset.event_id.duplicated().any():
            raise ValueError(f"Method {method} does not have exactly one row per held-out event.")

    # Reconstruct the unchanged A1 recommendations from the original event export.
    a1 = data["all_events"][
        (data["all_events"]["method"] == METHOD_A1)
        & (data["all_events"]["event_id"].isin(event_ids))
    ][["event_id", "hit10", "ndcg10", "top10_movie_ids", "top10_group_ids"]].copy()
    a1 = a1.merge(
        data["mapping"][["event_id", "conv_id"]],
        on="event_id", how="left", validate="one_to_one"
    )
    if len(a1) != len(event_ids) or a1.conv_id.isna().any():
        raise ValueError("Cannot reconstruct complete A1 held-out reference.")
    if set(a1.event_id.astype(int)) != set(event_ids):
        raise ValueError("A1 and policy event coverage differs.")

    method_frames = {METHOD_A1: a1}
    for method in POLICY_METHODS:
        method_frames[method] = policy_df[policy_df.method == method].copy()

    # Stable cell definition across all methods: verified categories x five groups.
    categories = sorted({
        cat for cats in data["item_categories"].values() for cat in cats
    })
    n_categories = len(categories)
    if n_categories == 0:
        raise ValueError("No accepted categories available for fairness inference.")
    n_cells = n_categories * 5

    conversations = np.asarray(sorted(set(conv_map.values())), dtype=np.int64)
    conv_index = {int(c): i for i, c in enumerate(conversations)}
    n_conv = len(conversations)
    conv_event_counts = np.zeros(n_conv, dtype=np.float64)

    # Event-level opportunity is common to every ranking method.
    event_opportunity = {}
    for event_id in event_ids:
        cr = data["candidate_by_event"][event_id]
        _, opp, _ = build_event_contribution(
            [], [], data["item_categories"], cr["candidate_ids"],
            data["item2group"]
        )
        event_opportunity[event_id] = opp
        conv_event_counts[conv_index[conv_map[event_id]]] += 1.0

    conv_opportunity = np.zeros((n_conv, n_cells), dtype=np.float64)
    for event_id in event_ids:
        conv_opportunity[conv_index[conv_map[event_id]]] += event_opportunity[event_id]

    # Per-method conversation-level exposure and utility sufficient statistics.
    conv_exposure = {}
    conv_hit = {}
    conv_ndcg = {}
    point_metrics = []

    for method, frame in method_frames.items():
        exp_by_conv = np.zeros((n_conv, n_cells), dtype=np.float64)
        hit_by_conv = np.zeros(n_conv, dtype=np.float64)
        ndcg_by_conv = np.zeros(n_conv, dtype=np.float64)

        records = frame.set_index("event_id").to_dict("index")
        if set(map(int, records)) != set(event_ids):
            raise ValueError(f"{method}: event IDs differ from held-out partition.")

        for event_id in event_ids:
            row = records[event_id]
            ids = list(map(int, M.read_json_list(row["top10_movie_ids"], "top10_movie_ids")))
            groups = list(map(int, M.read_json_list(row["top10_group_ids"], "top10_group_ids")))
            if len(ids) != M.TOP_K or len(groups) != M.TOP_K:
                raise ValueError(f"{method}: malformed Top-10 at event {event_id}.")
            exp, _, _ = build_event_contribution(
                ids, groups, data["item_categories"], [],
                data["item2group"]
            )
            ci = conv_index[conv_map[event_id]]
            exp_by_conv[ci] += exp
            hit_by_conv[ci] += float(row["hit10"])
            ndcg_by_conv[ci] += float(row["ndcg10"])

        conv_exposure[method] = exp_by_conv
        conv_hit[method] = hit_by_conv
        conv_ndcg[method] = ndcg_by_conv

        point_exp = exp_by_conv.sum(axis=0)[None, :]
        point_opp = conv_opportunity.sum(axis=0)[None, :]
        point_fairness = float(fairness_from_totals(
            point_exp, point_opp, n_categories
        )[0])
        point_metrics.append({
            "method": method,
            "events": len(event_ids),
            "conversations": n_conv,
            "HR@10": float(hit_by_conv.sum() / conv_event_counts.sum()),
            "NDCG@10": float(ndcg_by_conv.sum() / conv_event_counts.sum()),
            "category_group_log_rate_MAD": point_fairness,
        })

    # Paired cluster bootstrap. A multinomial count for each conversation
    # exactly preserves replacement multiplicity without expanding duplicate rows.
    rng = np.random.default_rng(args.seed)
    probabilities = np.full(n_conv, 1.0 / n_conv, dtype=np.float64)
    sample_counts = rng.multinomial(n_conv, probabilities, size=args.bootstrap)
    sampled_event_n = sample_counts @ conv_event_counts
    if np.any(sampled_event_n <= 0):
        raise RuntimeError("Bootstrap produced a replicate with no events.")

    # Recompute category disparity and paired utility deltas for each policy.
    a1_exp_boot = sample_counts @ conv_exposure[METHOD_A1]
    opp_boot = sample_counts @ conv_opportunity
    a1_fair_boot = fairness_from_totals(a1_exp_boot, opp_boot, n_categories)
    a1_hit_boot = (sample_counts @ conv_hit[METHOD_A1]) / sampled_event_n
    a1_ndcg_boot = (sample_counts @ conv_ndcg[METHOD_A1]) / sampled_event_n

    fairness_rows = []
    utility_rows = []
    bootstrap_point_rows = []

    for method in POLICY_METHODS:
        exp_boot = sample_counts @ conv_exposure[method]
        fair_boot = fairness_from_totals(exp_boot, opp_boot, n_categories)
        delta_fair = fair_boot - a1_fair_boot

        hit_boot = (sample_counts @ conv_hit[method]) / sampled_event_n
        ndcg_boot = (sample_counts @ conv_ndcg[method]) / sampled_event_n
        delta_hit = hit_boot - a1_hit_boot
        delta_ndcg = ndcg_boot - a1_ndcg_boot

        valid_fair = delta_fair[np.isfinite(delta_fair)]
        p_fair = 2.0 * min(
            float(np.mean(valid_fair <= 0.0)),
            float(np.mean(valid_fair >= 0.0))
        )
        p_fair = min(1.0, p_fair)

        fairness_rows.append({
            "method": method,
            "A1_fairness_point": float(
                next(x["category_group_log_rate_MAD"] for x in point_metrics if x["method"] == METHOD_A1)
            ),
            "method_fairness_point": float(
                next(x["category_group_log_rate_MAD"] for x in point_metrics if x["method"] == method)
            ),
            "delta_fairness_method_minus_A1": float(
                next(x["category_group_log_rate_MAD"] for x in point_metrics if x["method"] == method)
                - next(x["category_group_log_rate_MAD"] for x in point_metrics if x["method"] == METHOD_A1)
            ),
            "paired_delta_fairness_ci95_low": float(np.nanquantile(delta_fair, 0.025)),
            "paired_delta_fairness_ci95_high": float(np.nanquantile(delta_fair, 0.975)),
            "bootstrap_probability_fairness_improves": float(np.mean(delta_fair < 0.0)),
            "two_sided_sign_tail_probability": p_fair,
            "bootstrap_replicates": int(args.bootstrap),
            "conversations": int(n_conv),
        })
        utility_rows.append({
            "method": method,
            "delta_HR@10": float(
                next(x["HR@10"] for x in point_metrics if x["method"] == method)
                - next(x["HR@10"] for x in point_metrics if x["method"] == METHOD_A1)
            ),
            "delta_HR@10_ci95_low": float(np.quantile(delta_hit, 0.025)),
            "delta_HR@10_ci95_high": float(np.quantile(delta_hit, 0.975)),
            "delta_NDCG@10": float(
                next(x["NDCG@10"] for x in point_metrics if x["method"] == method)
                - next(x["NDCG@10"] for x in point_metrics if x["method"] == METHOD_A1)
            ),
            "delta_NDCG@10_ci95_low": float(np.quantile(delta_ndcg, 0.025)),
            "delta_NDCG@10_ci95_high": float(np.quantile(delta_ndcg, 0.975)),
            "bootstrap_replicates": int(args.bootstrap),
            "conversations": int(n_conv),
        })
        bootstrap_point_rows.append({
            "method": method,
            "fairness_delta_bootstrap": ci_summary(delta_fair),
            "HR_delta_bootstrap": ci_summary(delta_hit),
            "NDCG_delta_bootstrap": ci_summary(delta_ndcg),
        })

    pd.DataFrame(point_metrics).to_csv(args.out / "point_estimates.csv", index=False)
    pd.DataFrame(fairness_rows).to_csv(
        args.out / "paired_fairness_bootstrap_vs_A1.csv", index=False
    )
    pd.DataFrame(utility_rows).to_csv(
        args.out / "paired_utility_bootstrap_vs_A1.csv", index=False
    )
    (args.out / "bootstrap_distribution_summary.json").write_text(
        json.dumps(bootstrap_point_rows, indent=2), encoding="utf-8"
    )

    checks = {
        "experiment": "22.3_paired_conversation_cluster_fairness_inference",
        "validation_only": True,
        "test_split_evaluated": False,
        "events": len(event_ids),
        "conversations": n_conv,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "same_bootstrap_samples_for_all_methods": True,
        "sampling_unit": "conversation_id",
        "sampling_with_replacement": True,
        "duplicate_conversation_multiplicity_preserved": True,
        "fairness_metric": "category_group_log_rate_MAD, recomputed from aggregate sampled exposure/opportunity",
        "utility_metrics": ["HR@10", "NDCG@10"],
        "comparison_reference": METHOD_A1,
        "methods": POLICY_METHODS,
        "verified_category_count": n_categories,
        "category_group_cells": n_cells,
        "limitations": [
            "This is uncertainty estimation on the existing held-out validation partition, not independent test evaluation.",
            "The category-group disparity metric is the operational metric used in Experiments 22.0-22.2.",
            "The opportunity denominator is the normalized rank-discounted M2 Top-500 candidate-pool proxy.",
            "The bootstrap does not account for uncertainty in external category labels or KG entity alignment."
        ],
    }
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": "22.3_paired_conversation_cluster_fairness_inference",
        "point_estimates": point_metrics,
        "paired_fairness_inference": fairness_rows,
        "paired_utility_inference": utility_rows,
        "checks": checks,
    }
    (args.out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 22.3 COMPLETE")
    print("=" * 78)
    print("\nPoint estimates:")
    print(pd.DataFrame(point_metrics).to_string(index=False))
    print("\nPaired fairness bootstrap versus A1 (negative delta is better):")
    print(pd.DataFrame(fairness_rows)[[
        "method", "delta_fairness_method_minus_A1",
        "paired_delta_fairness_ci95_low",
        "paired_delta_fairness_ci95_high",
        "bootstrap_probability_fairness_improves"
    ]].to_string(index=False))
    print("\nPaired utility bootstrap versus A1:")
    print(pd.DataFrame(utility_rows)[[
        "method", "delta_HR@10", "delta_HR@10_ci95_low",
        "delta_HR@10_ci95_high", "delta_NDCG@10",
        "delta_NDCG@10_ci95_low", "delta_NDCG@10_ci95_high"
    ]].to_string(index=False))
    print("\nOutputs:", args.out.resolve())
    print("Test split evaluated: NO")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.3 FAILED: {exc}", file=sys.stderr)
        raise
