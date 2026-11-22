#!/usr/bin/env python
"""
Experiment 22.24C — Paired conversation-cluster inference on independent test.

Uses frozen 22.24B event-level rankings and the original frozen KBRD Top-500
candidate pools. Utility deltas are paired by event. Fairness MAD is recomputed
for each cluster-bootstrap replicate from multiplicity-weighted event-level
exposure and opportunity contributions, using the exact 22.16 definitions.

No training, retrieval, parameter search, or ranking changes are performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path("results_final_test_m2/experiment_22_24B")
DEFAULT_KBRD = Path("results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv")
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_OUT = Path("results_final_test_m2/experiment_22_24C")
N_BOOT = 10000
SEED = 20261115
EPS = 1e-12
MIN_CELL_CONVERSATIONS = 20


def load_reference(path):
    spec = importlib.util.spec_from_file_location("exp_22_16_for_22_24C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reference implementation: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("load_metadata", "ExposureLedger", "fairness_mad"):
        if not hasattr(mod, name):
            raise RuntimeError(f"Reference script missing {name}")
    return mod


def holm_adjust(p_values):
    ordered = sorted(p_values.items(), key=lambda x: x[1])
    adjusted, running = {}, 0.0
    m = len(ordered)
    for i, (name, p) in enumerate(ordered):
        value = min(1.0, (m - i) * float(p))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def two_sided_bootstrap_p(values, observed_delta):
    """
    Approximate a two-sided null test from the centered bootstrap
    estimation-error distribution.

    The bootstrap errors are centered at the observed effect:
        error_b = bootstrap_effect_b - observed_effect

    The two-sided tail probability compares |error_b| with
    |observed_effect|. The +1 correction avoids a zero Monte Carlo p-value.
    """
    values = np.asarray(values, dtype=float)
    observed_delta = float(observed_delta)
    errors = values - observed_delta
    extreme = np.count_nonzero(
        np.abs(errors) >= abs(observed_delta)
    )
    return float((extreme + 1) / (len(values) + 1))


def mad_from_totals(exposure, opportunity, cell_mask):
    """Compute log exposure/opportunity MAD over one fixed cell universe."""
    exposure = np.asarray(exposure, dtype=np.float64)
    opportunity = np.asarray(opportunity, dtype=np.float64)
    cell_mask = np.asarray(cell_mask, dtype=bool)

    if exposure.shape != opportunity.shape or cell_mask.shape != opportunity.shape:
        raise ValueError("Exposure, opportunity and cell-mask shapes must match.")
    if int(cell_mask.sum()) < 2:
        raise ValueError("At least two fairness cells are required.")
    if np.any(opportunity[cell_mask] <= EPS):
        raise RuntimeError("A selected fairness cell has zero bootstrap opportunity.")

    rates = (exposure[cell_mask] + EPS) / (opportunity[cell_mask] + EPS)
    logs = np.log(rates)
    return float(np.mean(np.abs(logs - np.mean(logs))))


def parse_ids(value):
    import ast
    try:
        arr = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        arr = json.loads(value)
    return [int(x) for x in arr]


def build_event_contributions(kbrd, result_frame, item2group, categories, cells, ref):
    """Return event-wise utility and fairness sufficient statistics."""
    kbrd_by_id = kbrd.set_index("event_id")
    result_by_id = result_frame.set_index("event_id")
    ids = sorted(result_by_id.index.astype(int))
    n_cells = len(cells)
    cell_index = {cell: i for i, cell in enumerate(cells)}
    exposure = np.zeros((len(ids), n_cells), dtype=np.float64)
    opportunity = np.zeros((len(ids), n_cells), dtype=np.float64)
    hits = np.zeros(len(ids), dtype=np.float64)
    ndcgs = np.zeros(len(ids), dtype=np.float64)
    convs = np.zeros(len(ids), dtype=np.int64)

    for i, event_id in enumerate(ids):
        rr = result_by_id.loc[event_id]
        kr = kbrd_by_id.loc[event_id]
        if int(rr.conv_id) != int(kr.conv_id) or int(rr.target_movie_id) != int(kr.target_movie_id):
            raise ValueError(f"Event identity mismatch for event_id={event_id}")

        candidate_ids = np.asarray(parse_ids(kr.top_movie_ids), dtype=np.int64)
        candidate_groups = np.asarray([item2group[int(x)] for x in candidate_ids], dtype=np.int64)
        ranked_ids = np.asarray(parse_ids(rr.top10_movie_ids), dtype=np.int64)
        ranked_groups = np.asarray(parse_ids(rr.top10_group_ids), dtype=np.int64)
        if len(candidate_ids) != 500 or len(ranked_ids) != 10 or len(ranked_groups) != 10:
            raise ValueError(f"Unexpected candidate/top-10 array size at event {event_id}")
        expected_groups = np.asarray([item2group[int(x)] for x in ranked_ids], dtype=np.int64)
        if not np.array_equal(expected_groups, ranked_groups):
            raise ValueError(f"Top-10 popularity-group mismatch at event {event_id}")

        # Exact 22.16 opportunity definition: normalized DCG weights over Top-500,
        # split evenly among each item's verified categories.
        opp_weights = ref.ExposureLedger.opportunity_weights(len(candidate_ids))
        for rank, movie_id in enumerate(candidate_ids):
            cats = categories.get(int(movie_id), [])
            if not cats:
                continue
            contribution = float(opp_weights[rank]) / len(cats)
            cell_group = int(candidate_groups[rank])
            for category in cats:
                opportunity[i, cell_index[(category, cell_group)]] += contribution

        # Exact 22.16 exposure definition: DCG exposure over the ranked Top-10.
        for rank, movie_id in enumerate(ranked_ids):
            cats = categories.get(int(movie_id), [])
            if not cats:
                continue
            contribution = 1.0 / math.log2(rank + 2.0) / len(cats)
            group = int(ranked_groups[rank])
            for category in cats:
                exposure[i, cell_index[(category, group)]] += contribution

        hits[i] = float(rr.hit10)
        ndcgs[i] = float(rr.ndcg10)
        convs[i] = int(rr.conv_id)

    return ids, convs, hits, ndcgs, exposure, opportunity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    ap.add_argument("--catalog-pickle", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--category-profiles", type=Path, default=DEFAULT_PROFILES)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--replicates", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--minimum-cell-conversations",
        type=int,
        default=MIN_CELL_CONVERSATIONS,
        help="Minimum distinct conversations supporting a fairness cell.",
    )
    args = ap.parse_args()

    baseline_path = args.results_dir / "baseline_event_results.csv"
    proposed_path = args.results_dir / "proposed_event_results.csv"
    for path in (baseline_path, proposed_path, args.kbrd, args.catalog_pickle,
                 args.category_profiles, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.replicates < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates; default is 10,000.")
    if args.minimum_cell_conversations < 2:
        raise ValueError("Minimum cell conversation support must be at least 2.")

    print("=" * 76)
    print("EXPERIMENT 22.24C — INDEPENDENT-TEST STATISTICAL INFERENCE")
    print("=" * 76)
    ref = load_reference(args.reference)
    item2group, catalog, categories = ref.load_metadata(
        args.catalog_pickle, args.category_profiles
    )

    baseline = pd.read_csv(baseline_path)
    proposed = pd.read_csv(proposed_path)
    kbrd = pd.read_csv(args.kbrd)

    required = {"event_id", "conv_id", "target_movie_id", "hit10", "ndcg10",
                "top10_movie_ids", "top10_group_ids"}
    for name, frame in (("baseline", baseline), ("proposed", proposed)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} results missing columns: {sorted(missing)}")
        if frame.event_id.duplicated().any():
            raise ValueError(f"Duplicate event IDs in {name} results.")

    kbrd_required = {"event_id", "conv_id", "target_movie_id", "top_movie_ids"}
    if kbrd_required - set(kbrd.columns):
        raise ValueError(f"KBRD export missing: {sorted(kbrd_required - set(kbrd.columns))}")
    if kbrd.event_id.duplicated().any():
        raise ValueError("Duplicate KBRD event IDs.")

    baseline = baseline.sort_values("event_id").reset_index(drop=True)
    proposed = proposed.sort_values("event_id").reset_index(drop=True)
    kbrd = kbrd.sort_values("event_id").reset_index(drop=True)
    if not (baseline.event_id.equals(proposed.event_id)
            and baseline.event_id.equals(kbrd.event_id)):
        raise ValueError("Baseline, proposed and KBRD event universes differ.")
    if len(baseline) != 4003 or baseline.conv_id.nunique() != 1283:
        raise ValueError("Expected exactly 4,003 events and 1,283 conversations.")
    for frame in (baseline, proposed):
        if not np.array_equal(frame.conv_id.to_numpy(), kbrd.conv_id.to_numpy()):
            raise ValueError("Conversation IDs differ from KBRD event manifest.")
        if not np.array_equal(frame.target_movie_id.to_numpy(), kbrd.target_movie_id.to_numpy()):
            raise ValueError("Target IDs differ from KBRD event manifest.")

    # Establish the full category x popularity-group cell universe from all
    # candidate opportunity contributions, not just observed top-10 exposure.
    all_cells = set()
    for value in kbrd.top_movie_ids:
        for movie_id in parse_ids(value):
            group = int(item2group[int(movie_id)])
            for category in categories.get(int(movie_id), []):
                all_cells.add((category, group))
    cells = sorted(all_cells)
    if len(cells) < 2:
        raise RuntimeError("Insufficient supported category-group cells.")

    print(f"Events={len(baseline)} | Conversations={baseline.conv_id.nunique()} | Cells={len(cells)}")
    print(f"Running paired conversation-cluster bootstrap ({args.replicates:,} replicates)...")

    b_ids, convs, bh, bn, be, bo = build_event_contributions(
        kbrd, baseline, item2group, categories, cells, ref
    )
    p_ids, p_convs, ph, pn, pe, po = build_event_contributions(
        kbrd, proposed, item2group, categories, cells, ref
    )
    if b_ids != p_ids or not np.array_equal(convs, p_convs):
        raise RuntimeError("Paired event/conversation alignment failed.")
    if not np.allclose(bo, po):
        raise RuntimeError("Baseline and proposed opportunity contributions differ.")

    # Utility point estimates use all paired independent-test events.
    base_hr, prop_hr = float(bh.mean()), float(ph.mean())
    base_ndcg, prop_ndcg = float(bn.mean()), float(pn.mean())

    unique_convs = np.unique(convs)
    conv_to_idx = {int(c): i for i, c in enumerate(unique_convs)}
    event_conv_idx = np.asarray([conv_to_idx[int(c)] for c in convs], dtype=np.int64)
    n_conv = len(unique_convs)
    n_events = len(convs)

    # Collapse event-level sufficient statistics to conversation clusters.
    def cluster_sum(values):
        result = np.zeros((n_conv,) + values.shape[1:], dtype=np.float64)
        np.add.at(result, event_conv_idx, values)
        return result

    cluster_count = np.bincount(event_conv_idx, minlength=n_conv).astype(np.float64)
    cluster_bh, cluster_ph = cluster_sum(bh[:, None])[:, 0], cluster_sum(ph[:, None])[:, 0]
    cluster_bn, cluster_pn = cluster_sum(bn[:, None])[:, 0], cluster_sum(pn[:, None])[:, 0]
    cluster_be, cluster_pe = cluster_sum(be), cluster_sum(pe)
    cluster_opp = cluster_sum(bo)
    
# -------------------------------------------------------
# FAIRNESS BOOTSTRAP SUPPORT DIAGNOSTIC
# -------------------------------------------------------

    cell_support = (cluster_opp > EPS).sum(axis=0)
    fairness_cell_mask = cell_support >= args.minimum_cell_conversations

    if int(fairness_cell_mask.sum()) < 2:
        raise RuntimeError(
            "Fewer than two category-group cells meet the minimum support threshold."
        )

    support_df = pd.DataFrame({
        "category": [c[0] for c in cells],
        "popularity_group": [c[1] for c in cells],
        "supporting_conversations": cell_support,
        "total_conversations": n_conv,
        "support_percentage": 100.0 * cell_support / n_conv,
        "minimum_support_threshold": args.minimum_cell_conversations,
        "included_for_inference": fairness_cell_mask,
    })

    print("\n========== FIXED FAIRNESS CELL UNIVERSE ==========")
    print("Total category-group cells:", len(cells))
    print("Minimum supporting conversations:", int(cell_support.min()))
    print("Median supporting conversations:", float(np.median(cell_support)))
    print("Maximum supporting conversations:", int(cell_support.max()))
    print("Minimum support threshold:", args.minimum_cell_conversations)
    print("Included cells:", int(fairness_cell_mask.sum()))
    print("Excluded cells:", int((~fairness_cell_mask).sum()))

    args.out.mkdir(parents=True, exist_ok=True)
    support_df.to_csv(args.out / "fairness_cell_support.csv", index=False)
    print("Saved fairness_cell_support.csv")

    # Both systems use the same observed opportunity denominator and fixed cells.
    total_opp = bo.sum(axis=0)
    base_mad = mad_from_totals(be.sum(axis=0), total_opp, fairness_cell_mask)
    prop_mad = mad_from_totals(pe.sum(axis=0), total_opp, fairness_cell_mask)


    rng = np.random.default_rng(args.seed)
    delta_hr = np.empty(args.replicates)
    delta_ndcg = np.empty(args.replicates)
    delta_mad = np.empty(args.replicates)

    for start in range(0, args.replicates, 250):
        stop = min(start + 250, args.replicates)
        m = stop - start
        sampled = rng.integers(0, n_conv, size=(m, n_conv))
        multiplicity = np.zeros((m, n_conv), dtype=np.float64)
        rows = np.repeat(np.arange(m), n_conv)
        np.add.at(multiplicity, (rows, sampled.ravel()), 1.0)
        denom = multiplicity @ cluster_count
        delta_hr[start:stop] = (
            multiplicity @ (cluster_ph - cluster_bh)
        ) / denom
        delta_ndcg[start:stop] = (
            multiplicity @ (cluster_pn - cluster_bn)
        ) / denom
        boot_be = multiplicity @ cluster_be
        boot_pe = multiplicity @ cluster_pe
        boot_opp = multiplicity @ cluster_opp
        for j in range(m):
            delta_mad[start + j] = (
                mad_from_totals(boot_pe[j], boot_opp[j], fairness_cell_mask)
                - mad_from_totals(boot_be[j], boot_opp[j], fairness_cell_mask)
            )

    observed = {
        "HR@10": prop_hr - base_hr,
        "NDCG@10": prop_ndcg - base_ndcg,
        "fairness_MAD": prop_mad - base_mad,
    }
    distributions = {
        "HR@10": delta_hr,
        "NDCG@10": delta_ndcg,
        "fairness_MAD": delta_mad,
    }
    # Test H0: treatment effect = 0 using the null-centered bootstrap
    # estimation-error distribution. Then apply Holm across all three metrics.
    pvals = {
        name: two_sided_bootstrap_p(distributions[name], observed[name])
        for name in distributions
    }
    adjusted = holm_adjust(pvals)

    bootstrap_diagnostics = {}
    for name, values in distributions.items():
        values = np.asarray(values, dtype=float)
        bootstrap_diagnostics[name] = {
            "observed_delta": float(observed[name]),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "bootstrap_bias": float(np.mean(values) - observed[name]),
            "bootstrap_std": float(np.std(values, ddof=1)),
            "bootstrap_percentile_q025": float(np.quantile(values, 0.025)),
            "bootstrap_percentile_q975": float(np.quantile(values, 0.975)),
            "basic_bootstrap_ci95_low": float(2.0 * observed[name] - np.quantile(values, 0.975)),
            "basic_bootstrap_ci95_high": float(2.0 * observed[name] - np.quantile(values, 0.025)),
            "fraction_negative": float(np.mean(values < 0)),
            "two_sided_p": float(pvals[name]),
            "holm_adjusted_p": float(adjusted[name]),
        }

    rows = []
    for name in ("HR@10", "NDCG@10", "fairness_MAD"):
        vals = distributions[name]
        q025 = float(np.quantile(vals, 0.025))
        q975 = float(np.quantile(vals, 0.975))
        ci_low = float(2.0 * observed[name] - q975)
        ci_high = float(2.0 * observed[name] - q025)
        rows.append({
            "metric": name,
            "baseline": base_hr if name == "HR@10" else base_ndcg if name == "NDCG@10" else base_mad,
            "proposed": prop_hr if name == "HR@10" else prop_ndcg if name == "NDCG@10" else prop_mad,
            "delta_proposed_minus_baseline": observed[name],
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_p_two_sided": pvals[name],
            "holm_adjusted_p": adjusted[name],
            "significant_after_holm": bool(adjusted[name] < 0.05),
        })
    inference = pd.DataFrame(rows)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    inference.to_csv(out / "paired_cluster_inference.csv", index=False)
    inference[["metric", "ci95_low", "ci95_high"]].to_csv(
        out / "confidence_intervals.csv", index=False
    )
    np.savez_compressed(
        out / "bootstrap_distributions.npz",
        delta_hr10=delta_hr, delta_ndcg10=delta_ndcg, delta_fairness_mad=delta_mad,
    )
    (out / "bootstrap_diagnostics.json").write_text(
        json.dumps(bootstrap_diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame.from_dict(bootstrap_diagnostics, orient="index").rename_axis(
        "metric"
    ).reset_index().to_csv(out / "bootstrap_diagnostics.csv", index=False)

    checks = {
        "exactly_4003_paired_events": len(baseline) == 4003,
        "exactly_1283_conversation_clusters": n_conv == 1283,
        "same_event_universe": b_ids == p_ids,
        "same_conversation_mapping": np.array_equal(convs, p_convs),
        "same_opportunity_universe_and_contributions": bool(np.allclose(bo, po)),
        "event_level_opportunity_recomputed_from_frozen_top500": True,
        "exposure_recomputed_from_saved_top10_rankings": True,
        "fixed_fairness_cell_universe": True,
        "minimum_cell_support_applied": bool(
            np.all(cell_support[fairness_cell_mask] >= args.minimum_cell_conversations)
        ),
        "fairness_mad_recomputed_per_bootstrap_replicate": True,
        "conversation_cluster_bootstrap": True,
        "bootstrap_replicates": args.replicates,
        "holm_correction_applied": True,
        "training_performed": False,
        "retrieval_rerun": False,
        "parameter_search": False,
    }
    required_true_checks = [
        "exactly_4003_paired_events",
        "exactly_1283_conversation_clusters",
        "same_event_universe",
        "same_conversation_mapping",
        "same_opportunity_universe_and_contributions",
        "event_level_opportunity_recomputed_from_frozen_top500",
        "exposure_recomputed_from_saved_top10_rankings",
        "fixed_fairness_cell_universe",
        "minimum_cell_support_applied",
        "fairness_mad_recomputed_per_bootstrap_replicate",
        "conversation_cluster_bootstrap",
        "holm_correction_applied",
    ]
    checks["overall_pass"] = (
        all(checks[key] is True for key in required_true_checks)
        and checks["training_performed"] is False
        and checks["retrieval_rerun"] is False
        and checks["parameter_search"] is False
    )
    checks["minimum_cell_conversations"] = args.minimum_cell_conversations
    checks["fairness_cells_total"] = len(cells)
    checks["fairness_cells_included"] = int(fairness_cell_mask.sum())
    checks["fairness_cells_excluded"] = int((~fairness_cell_mask).sum())

    retention = prop_ndcg / base_ndcg if base_ndcg > 0 else float("nan")
    assessment = {
        "experiment": "22.24C_independent_test_statistical_inference",
        "split": "independent_test",
        "events": len(baseline),
        "conversations": n_conv,
        "bootstrap_replicates": args.replicates,
        "bootstrap_unit": "conversation",
        "baseline_HR@10": base_hr,
        "proposed_HR@10": prop_hr,
        "baseline_NDCG@10": base_ndcg,
        "proposed_NDCG@10": prop_ndcg,
        "NDCG_retention": retention,
        "baseline_fairness_MAD": base_mad,
        "proposed_fairness_MAD": prop_mad,
        "fairness_delta": prop_mad - base_mad,
        "fairness_reduction_percent": 100.0 * (base_mad - prop_mad) / base_mad if base_mad > 0 else None,
        "fairness_significant_after_holm": bool(adjusted["fairness_MAD"] < 0.05),
        "utility_retained_at_95pct": bool(retention >= 0.95),
        "statistical_results": rows,
        "bootstrap_diagnostics": bootstrap_diagnostics,
        "p_value_method": (
            "Approximate two-sided null-centered bootstrap estimation-error tail probability "
            "with +1 correction; Holm correction across HR@10, NDCG@10 and fairness_MAD."
        ),
        "validation": checks,
        "confidence_interval_method": "Basic paired conversation-cluster bootstrap interval.",
        "fairness_estimand": (
            "Mean absolute deviation of log exposure/opportunity ratios over the fixed "
            "category-popularity cells supported by at least "
            f"{args.minimum_cell_conversations} distinct conversations."
        ),
        "interpretation_note": (
            "The fairness cell universe is fixed from observed opportunity support and "
            "applied identically to baseline, proposed and every bootstrap replicate. "
            "Rankings are the frozen sequential 22.24B rankings; this is conditional "
            "inference and does not rerun the adaptive ranking policy within bootstrap "
            "replicates."
        ),
    }
    (out / "validation_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (out / "final_research_assessment.json").write_text(
        json.dumps(assessment, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.24C COMPLETE ==========")
    print(inference.to_string(index=False))
    print("\nResearch assessment:")
    print(json.dumps(assessment, indent=2, allow_nan=False))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()

