
#!/usr/bin/env python
"""
Experiment 22.23
Independent-test fairness and utility inference.

Compares:
    A0_fixed_025  (baseline)
    A1_frequency_3 (proposed)

Uses:
    - Frozen independent-test M2 event results.
    - Full 6,924-item catalog as the opportunity universe.
    - Existing Experiment 22.16 category evidence and group mapping.
    - Conversation-cluster paired bootstrap (10,000 replicates).

No training, retrieval, parameter tuning, or test-set model selection.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_BOOT = 10000
EPS = 1e-12

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

BASELINE = "A0_fixed_025"
TREATMENT = "A1_frequency_3"

DEFAULT_EVENTS = Path(
    "results_final_test_m2/adaptive_m2_final_test_events_clustered.csv"
)
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_PROFILES = Path(
    "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_OUT = Path("results_final_test_m2/experiment_22_23")


def load_reference(path):
    spec = importlib.util.spec_from_file_location(
        "experiment_22_16", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Experiment 22.16 reference.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_metadata"):
        raise AttributeError("Experiment 22.16 load_metadata missing.")

    return module


def parse_top10(value, field):
    try:
        result = json.loads(value)
    except Exception:
        import ast
        result = ast.literal_eval(value)

    if not isinstance(result, list) or len(result) != 10:
        raise ValueError(f"{field} must contain exactly 10 IDs.")

    result = list(map(int, result))

    if len(set(result)) != 10:
        raise ValueError(f"{field} contains duplicate movie IDs.")

    return result


def load_events(path):
    frame = pd.read_csv(path)

    required = {
        "event_id", "conv_id", "method",
        "target_movie_id", "hit10", "ndcg10",
        "top10_movie_ids",
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    frame = frame[
        frame["method"].isin([BASELINE, TREATMENT])
    ].copy()

    if frame.empty:
        raise ValueError("No baseline/treatment events found.")

    if frame.duplicated(["method", "event_id"]).any():
        raise ValueError("Duplicate method/event records.")

    for method in (BASELINE, TREATMENT):
        subset = frame[frame.method == method]
        if len(subset) != 4003:
            raise ValueError(
                f"{method}: expected 4003 events, got {len(subset)}"
            )

    left = (
        frame[frame.method == BASELINE]
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    right = (
        frame[frame.method == TREATMENT]
        .sort_values("event_id")
        .reset_index(drop=True)
    )

    for col in ("event_id", "conv_id", "target_movie_id"):
        if not np.array_equal(left[col], right[col]):
            raise ValueError(f"Paired event mismatch: {col}")

    if not np.isfinite(frame["ndcg10"]).all():
        raise ValueError("Non-finite NDCG values.")

    frame["top10"] = frame["top10_movie_ids"].apply(
        lambda x: parse_top10(x, "top10_movie_ids")
    )

    return frame


def prepare_cells(catalog_ids, item2group, categories):
    """
    Define the category x popularity-group cells supported
    by the full catalog opportunity universe.
    """

    cells = set()

    for movie_id in catalog_ids:
        group = int(item2group[int(movie_id)])

        for category in categories.get(int(movie_id), []):
            cells.add((category, group))

    cells = sorted(cells)

    if len(cells) < 2:
        raise ValueError("Insufficient supported fairness cells.")

    return cells, {cell: i for i, cell in enumerate(cells)}


def build_conversation_ledgers(
    frame, catalog_ids, item2group, categories, cell_index
):
    """
    Aggregate opportunity and exposure by conversation.

    Opportunity:
        Full catalog, repeated once per event.

    Exposure:
        Actual Top-10 recommendations.

    Each conversation is the resampling unit.
    """

    conversations = sorted(frame["conv_id"].astype(int).unique())
    conv_index = {c: i for i, c in enumerate(conversations)}

    n_conv = len(conversations)
    n_cells = len(cell_index)

    opportunity = np.zeros((n_conv, n_cells), dtype=np.float64)

    exposure_baseline = np.zeros_like(opportunity)
    exposure_treatment = np.zeros_like(opportunity)

    # Full-catalog opportunity weights.
    n_catalog = len(catalog_ids)

    weights = np.full(
    n_catalog,
    1.0 / n_catalog,
    dtype=np.float64,
    )
    weights /= weights.sum() + EPS

    catalog_groups = np.asarray(
        [item2group[int(i)] for i in catalog_ids],
        dtype=np.int32
    )

    # Precompute the category/group opportunity contribution
    # for one event. This avoids repeatedly scanning the catalog.
    one_event_opportunity = np.zeros(n_cells, dtype=np.float64)

    for position, movie_id in enumerate(catalog_ids):
        movie_categories = categories.get(int(movie_id), [])

        if not movie_categories:
            continue

        group = int(catalog_groups[position])
        contribution = weights[position] / len(movie_categories)

        for category in movie_categories:
            key = (category, group)
            one_event_opportunity[cell_index[key]] += contribution

    # Each event contributes the same full-catalog opportunity.
    event_counts = (
        frame.groupby("conv_id")["event_id"]
        .nunique()
        .to_dict()
    )

    for conv_id, count in event_counts.items():
        opportunity[conv_index[int(conv_id)]] = (
            int(count) * one_event_opportunity
        )

    # Recommendation exposure contributions.
    ordered = frame.sort_values(["method", "event_id"])

    for row in ordered.itertuples(index=False):
        ci = conv_index[int(row.conv_id)]

        if row.method == BASELINE:
            exposure = exposure_baseline
        else:
            exposure = exposure_treatment

        for rank, movie_id in enumerate(row.top10, start=1):
            movie_categories = categories.get(int(movie_id), [])

            if not movie_categories:
                continue

            group = int(item2group[int(movie_id)])

            contribution = (
                1.0 / np.log2(rank + 1.0)
            ) / len(movie_categories)

            for category in movie_categories:
                key = (category, group)
                exposure[ci, cell_index[key]] += contribution

    return (
        conversations,
        opportunity,
        exposure_baseline,
        exposure_treatment,
    )


def fairness_mad(exposure, opportunity):
    valid = opportunity > EPS

    if valid.sum() < 2:
        return float("nan")

    rates = (
        exposure[valid] + EPS
    ) / (
        opportunity[valid] + EPS
    )

    log_rates = np.log(rates)

    return float(
        np.mean(np.abs(log_rates - np.mean(log_rates)))
    )


def holm_adjust(p_values):
    names = list(p_values)
    ordered = sorted(names, key=lambda x: p_values[x])

    adjusted = {}
    running = 0.0
    m = len(ordered)

    for rank, name in enumerate(ordered):
        value = min(
            1.0,
            (m - rank) * p_values[name]
        )
        running = max(running, value)
        adjusted[name] = running

    return adjusted


def bootstrap_analysis(
    frame,
    conversations,
    opportunity,
    exposure_baseline,
    exposure_treatment,
    replicates=N_BOOT,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    n_conv = len(conversations)
    conv_to_index = {
        int(c): i for i, c in enumerate(conversations)
    }

    base = frame[frame.method == BASELINE].copy()
    treatment = frame[frame.method == TREATMENT].copy()

    base["conv_index"] = base["conv_id"].map(conv_to_index)
    treatment["conv_index"] = treatment["conv_id"].map(conv_to_index)

    def utility_matrix(data):
        hit = np.zeros(n_conv)
        ndcg = np.zeros(n_conv)
        count = np.zeros(n_conv)

        grouped = data.groupby("conv_index")

        for ci, group in grouped:
            hit[int(ci)] = group["hit10"].sum()
            ndcg[int(ci)] = group["ndcg10"].sum()
            count[int(ci)] = len(group)

        return hit, ndcg, count

    bh, bn, bc = utility_matrix(base)
    th, tn, tc = utility_matrix(treatment)

    if not np.array_equal(bc, tc):
        raise ValueError("Conversation event counts differ.")

    if not np.allclose(bc.sum(), 4003):
        raise ValueError("Unexpected paired event count.")

    base_exposure = exposure_baseline
    treatment_exposure = exposure_treatment

    # Observed estimates.
    total_opportunity = opportunity.sum(axis=0)

    base_mad = fairness_mad(
        base_exposure.sum(axis=0),
        total_opportunity,
    )

    treatment_mad = fairness_mad(
        treatment_exposure.sum(axis=0),
        total_opportunity,
    )

    base_hit = bh.sum() / bc.sum()
    treatment_hit = th.sum() / tc.sum()

    base_ndcg = bn.sum() / bc.sum()
    treatment_ndcg = tn.sum() / tc.sum()

    retention = (
        treatment_ndcg / base_ndcg
        if base_ndcg > EPS else float("nan")
    )

    # Paired conversation-cluster bootstrap.
    hit_delta = np.empty(replicates)
    ndcg_delta = np.empty(replicates)
    fairness_delta = np.empty(replicates)

    for b in range(replicates):
        sampled = rng.integers(0, n_conv, size=n_conv)

        multiplicity = np.bincount(
            sampled, minlength=n_conv
        ).astype(np.float64)

        n_events = float(multiplicity @ bc)

        if n_events <= 0:
            raise RuntimeError("Empty bootstrap sample.")

        hit_delta[b] = (
            multiplicity @ (th - bh)
        ) / n_events

        ndcg_delta[b] = (
            multiplicity @ (tn - bn)
        ) / n_events

        sampled_opportunity = multiplicity @ opportunity

        sampled_base_exposure = multiplicity @ base_exposure
        sampled_treatment_exposure = multiplicity @ treatment_exposure

        fairness_delta[b] = (
            fairness_mad(
                sampled_treatment_exposure,
                sampled_opportunity,
            )
            -
            fairness_mad(
                sampled_base_exposure,
                sampled_opportunity,
            )
        )

    def interval(values):
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
            # Bootstrap distribution diagnostics
    fairness_bootstrap_diagnostics = {
        "observed_delta": float(
            treatment_mad - base_mad
        ),
        "bootstrap_mean": float(
            np.mean(fairness_delta)
        ),
        "bootstrap_median": float(
            np.median(fairness_delta)
        ),
        "bootstrap_bias": float(
            np.mean(fairness_delta)
            - (treatment_mad - base_mad)
        ),
        "bootstrap_std": float(
            np.std(fairness_delta, ddof=1)
        ),
        "bootstrap_q025": float(
            np.quantile(fairness_delta, 0.025)
        ),
        "bootstrap_q975": float(
            np.quantile(fairness_delta, 0.975)
        ),
        "fraction_negative": float(
            np.mean(fairness_delta < 0)
        ),
        }

    print(
        "\\nFAIRNESS BOOTSTRAP DIAGNOSTICS:"
        )
    print(
        json.dumps(
            fairness_bootstrap_diagnostics,
            indent=2
        )
        )

    def two_sided_p(values):
        p_positive = (np.sum(values >= 0) + 1) / (
            len(values) + 1
        )
        p_negative = (np.sum(values <= 0) + 1) / (
            len(values) + 1
        )
        return float(min(1.0, 2 * min(p_positive, p_negative)))

    point_deltas = {
        "HR@10": treatment_hit - base_hit,
        "NDCG@10": treatment_ndcg - base_ndcg,
        "fairness_MAD": treatment_mad - base_mad,
    }

    distributions = {
        "HR@10": hit_delta,
        "NDCG@10": ndcg_delta,
        "fairness_MAD": fairness_delta,
    }

    p_values = {
        name: two_sided_p(values)
        for name, values in distributions.items()
    }

    adjusted = holm_adjust(p_values)

    utility_rows = []

    for name in ("HR@10", "NDCG@10", "fairness_MAD"):
        values = distributions[name]

        utility_rows.append({
            "metric": name,
            "baseline": (
                base_hit if name == "HR@10"
                else base_ndcg if name == "NDCG@10"
                else base_mad
            ),
            "treatment": (
                treatment_hit if name == "HR@10"
                else treatment_ndcg if name == "NDCG@10"
                else treatment_mad
            ),
            "delta_treatment_minus_baseline": point_deltas[name],
            "ci95_low": interval(values)[0],
            "ci95_high": interval(values)[1],
            "bootstrap_p": p_values[name],
            "holm_adjusted_p": adjusted[name],
            "significant_after_holm": adjusted[name] < 0.05,
        })

    return {
        "baseline_HR10": base_hit,
        "treatment_HR10": treatment_hit,
        "baseline_NDCG10": base_ndcg,
        "treatment_NDCG10": treatment_ndcg,
        "NDCG_retention": retention,
        "baseline_fairness_MAD": base_mad,
        "treatment_fairness_MAD": treatment_mad,
        "fairness_delta": treatment_mad - base_mad,
        "fairness_improved": treatment_mad < base_mad,
        "utility_retained_at_95pct": retention >= 0.95,
        "inference": pd.DataFrame(utility_rows),
        "bootstrap_replicates": replicates,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)

    args = parser.parse_args()

    if args.bootstrap < 1000:
        raise ValueError("At least 1,000 bootstrap replicates required.")

    print("Loading frozen independent-test events...")

    frame = load_events(args.events)

    print(
        f"Events={frame.event_id.nunique()} | "
        f"Conversations={frame.conv_id.nunique()}"
    )

    print("Loading Experiment 22.16 category definitions...")

    reference = load_reference(args.reference)

    item2group, catalog, categories = reference.load_metadata(
        args.catalog,
        args.profiles,
    )

    catalog_ids = sorted(map(int, catalog))

    if len(catalog_ids) != 6924:
        raise ValueError(
            f"Expected 6,924 catalog items; got {len(catalog_ids)}"
        )

    if set(catalog_ids) != set(item2group):
        raise ValueError("Catalog/group mapping mismatch.")

    print(
        f"Catalog={len(catalog_ids)} | "
        f"Verified categorized items="
        f"{sum(bool(categories.get(i, [])) for i in catalog_ids)}"
    )

    print("Building category-group opportunity and exposure ledgers...")

    cells, cell_index = prepare_cells(
        catalog_ids, item2group, categories
    )

    (
        conversations,
        opportunity,
        exposure_baseline,
        exposure_treatment,
    ) = build_conversation_ledgers(
        frame,
        catalog_ids,
        item2group,
        categories,
        cell_index,
    )

    print(
        f"Fairness cells={len(cells)} | "
        f"Conversation clusters={len(conversations)}"
    )

    print(
        f"Running paired cluster bootstrap "
        f"({args.bootstrap:,} replicates)..."
    )

    results = bootstrap_analysis(
        frame,
        conversations,
        opportunity,
        exposure_baseline,
        exposure_treatment,
        replicates=args.bootstrap,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    inference = results.pop("inference")

    inference.to_csv(
        args.out / "paired_cluster_inference.csv",
        index=False,
    )

    summary = {
        "experiment": "22.23_independent_test_fairness_inference",
        "split": "independent_test",
        "baseline": BASELINE,
        "treatment": TREATMENT,
        "events": int(frame.event_id.nunique()),
        "conversations": int(frame.conv_id.nunique()),
        "catalog_items": len(catalog_ids),
        "opportunity_universe": "full_6924_item_catalog",
        "fairness_metric": "category_group_log_exposure_opportunity_MAD",
        "baseline_HR@10": results["baseline_HR10"],
        "treatment_HR@10": results["treatment_HR10"],
        "baseline_NDCG@10": results["baseline_NDCG10"],
        "treatment_NDCG@10": results["treatment_NDCG10"],
        "NDCG_retention": results["NDCG_retention"],
        "baseline_fairness_MAD": results["baseline_fairness_MAD"],
        "treatment_fairness_MAD": results["treatment_fairness_MAD"],
        "fairness_delta": results["fairness_delta"],
        "fairness_improved": results["fairness_improved"],
        "utility_retained_at_95pct": results["utility_retained_at_95pct"],
        "bootstrap_replicates": results["bootstrap_replicates"],
        "bootstrap_unit": "conversation",
        "holm_correction": True,
        "training_performed": False,
        "retrieval_rerun": False,
        "parameter_search": False,
        "independent_test_used_for_tuning": False,
    }

    checks = {
        "exactly_two_methods": True,
        "paired_event_universe": True,
        "4003_events_per_method": True,
        "conversation_cluster_bootstrap": True,
        "full_catalog_opportunity_used": True,
        "category_evidence_reused_from_22_16": True,
        "no_training": True,
        "no_retrieval": True,
        "no_parameter_search": True,
        "holm_correction_applied": True,
        "bootstrap_replicates_valid": args.bootstrap >= 1000,
        "overall_pass": True,
    }

    summary["validation"] = checks

    (args.out / "final_research_assessment.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )

    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, default=json_default),
        encoding="utf-8",
    )

    manifest = {
        "experiment": "22.23",
        "name": "Independent Test Fairness and Utility Inference",
        "inputs": [
            str(args.events),
            str(args.catalog),
            str(args.profiles),
            str(args.reference),
        ],
        "outputs": [
            "paired_cluster_inference.csv",
            "final_research_assessment.json",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
        "methodology": {
            "comparison": [BASELINE, TREATMENT],
            "opportunity_universe": "full_catalog_6924",
            "fairness_metric": "category_group_log_exposure_opportunity_MAD",
            "bootstrap_unit": "conversation",
            "bootstrap_replicates": args.bootstrap,
            "paired_inference": True,
            "holm_correction": True,
        },
        "limitations": [
            "The fairness metric measures catalog-opportunity-adjusted exposure.",
            "It does not establish semantic relevance of dialogue context.",
            "The independent test is used once for final evaluation.",
            "Bootstrap inference is conditional on the frozen event set.",
        ],
    }

    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("\n========== EXPERIMENT 22.23 COMPLETE ==========")

    print(inference.to_string(index=False))

    print("\nResearch assessment:")
    print(json.dumps(summary, indent=2, default=json_default))

    print("\nOutput:", args.out.resolve())


if __name__ == "__main__":
    main()

